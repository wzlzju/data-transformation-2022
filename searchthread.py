import json, csv
import os, sys, time
from copy import deepcopy
import pandas as pd

from treelib import Tree, Node
from multiprocessing import Process, Pool, Queue
from _queue import Empty
from queue import Full

from spreadsheet import spreadsheet
from V import *
from T import *
from Tfunctions import *
from L import *
from config import *
from utils import *

def tpaththreadfunction(tname, colinfo, q):
    if RANKINGON:
        colinfo, rank_tp = ranking(colinfo)
    t = tlist[tname]
    tinput = t["input"]
    tinputdim = tinput["dim"] if isinstance(tinput, dict) else None
    tinputtype = tinput["type"] if isinstance(tinput, dict) else None
    colnames = colinfo["col_names"]
    numcolnames = colinfo["num_col_names"]
    coltype = colinfo["col_type"]
    distmat = colinfo["dist_mat"]
    colnamessimi = colinfo["col_names_simi"]
    colnamesvectors = colnamessimi["vectors"]
    simimat = colnamessimi["cosine"]
    pool = []

    def updatequeue():
        sleep_time = 0.1
        if not q:
            return
        if not q.full():
            try:
                q.put(pool, block=True, timeout=sleep_time)
            except Full as e:
                pass
        else:
            try:
                q.get(block=True, timeout=sleep_time)
            except Empty as e:
                pass
            try:
                q.put(pool, block=True, timeout=sleep_time)
            except Full as e:
                pass

    if tname == "test":
        pool.append((0, tpath()))
        tpp = pool[0][1]
        tpp.append({
            "t": "sum",
            "i_type": "like",
            "i": ["int", "float"],
            "o_type": "new_table",
            "args": (),
            "kwargs": {"axis": 1},
            "index": "default"
        })
        # tpp.append({
        #     "t": "astype",
        #     "i_type": "like",
        #     "i": ["int", "float"],
        #     "o_type": "new_table",
        #     "args": ("float64", ),
        #     "kwargs": {},
        #     "index": "default"
        # })
        if MULTIPROCESS:
            while True:
                updatequeue()
    # elif tname == "pca":
    #     pool.append((0, tpath()))
    #     tpp = pool[0][1]
    #     while True:
    #         updatequeue()
    # elif tname == "lda":
    #     pool.append((0, tpath()))
    #     tpp = pool[0][1]
    #     while True:
    #         updatequeue()

    if tname == "null_nom1":
        for col in colnames:
            if coltype[col]["type"] == "nominal" and not coltype[col]["iskey"]:
                # pool.append((0, tpath([{
                #     "t": "nominalize",
                #     "i_type": "==",
                #     "i": [col],
                #     "o_type": "new_table",
                #     "args": (),
                #     "kwargs": {"axis": 1},
                #     "index": pd.Index([col, "NOMINAL "+col])
                # }])))
                pool.append((0, tpath([{
                    "t": "select",
                    "i_type": "==",
                    "i": [col],
                    "o_type": "new_table",
                    "args": (),
                    "kwargs": {},
                    "index": "default"
                }])))
                if MULTIPROCESS:
                    updatequeue()
    elif tname == "null_nom":
        noml = []
        for col in colnames:
            if coltype[col]["type"] == "nominal" and not coltype[col]["iskey"]:
                noml.append(col)
        pool.append((0, tpath([{
            "t": "select",
            "i_type": "==",
            "i": noml,
            "o_type": "new_table",
            "args": (),
            "kwargs": {},
            "index": "default"
        }])))
        if MULTIPROCESS:
            updatequeue()
    elif tname == "null_num1":
        for col in colnames:
            if coltype[col]["type"] in ["real", "int"] and not coltype[col]["iskey"]:
                tpp = tpath()
                if hasRANK(col):
                    tpp.append(rank_tp)
                tpp.append({
                    "t": "select",
                    "i_type": "==",
                    "i": [col],
                    "o_type": "new_table",
                    "args": (),
                    "kwargs": {},
                    "index": "default"
                })
                pool.append((0, tpp))
                if MULTIPROCESS:
                    updatequeue()
    elif tinputtype == "num":
        # e.g., pca, lda, kmeans

        # depth 0
        cur_columns = [colname for colname in colnames if coltype[colname]["type"] in ["int", "real"]]

        # dimension matching
        clusters = colinfo["dim_match"]["clusters"]
        print("in thread", tname, clusters)
        for i, cluster in enumerate(clusters):
            cur_cluster = listintersection(cluster, cur_columns)
            if len(cur_cluster) > 0:
                tpp = tpath()
                if hasRANK(cur_cluster):
                    tpp.append(rank_tp)
                tpp.append({
                    "t": "select",
                    "i_type": "==",
                    "i": cur_cluster,
                    "o_type": "new_table",
                    "args": (),
                    "kwargs": {},
                    "index": "default"
                })
                pool.append((0, tpp))
                if MULTIPROCESS:
                    updatequeue()
        # semantic matching
        clusters = colinfo["col_names_simi"]["clusters"]
        print("in thread", tname, clusters)
        for i, cluster in enumerate(clusters):
            cur_cluster = listintersection(cluster, cur_columns)
            if len(cur_cluster) > 0:
                tpp = tpath()
                if hasRANK(cur_cluster):
                    tpp.append(rank_tp)
                tpp.append({
                    "t": "select",
                    "i_type": "==",
                    "i": cur_cluster,
                    "o_type": "new_table",
                    "args": (),
                    "kwargs": {},
                    "index": "default"
                })
                pool.append((0, tpp))
                if MULTIPROCESS:
                    updatequeue()

        # depth >= 1
        num_dim_clusters = [listintersection(cluster, numcolnames) for cluster in colinfo["dim_match"]["clusters"]]
        num_sem_clusters = [listintersection(cluster, numcolnames) for cluster in colinfo["col_names_simi"]["clusters"]]
        num_clusters = [iii for iii in num_dim_clusters] + [jjj for jjj in num_sem_clusters if jjj not in num_dim_clusters]
        tree = Tree()
        tree.create_node(tag='root', identifier='root', data={
            'colinfo': colinfo,
            'load': (0, 0, 0),
            'tpath': tpath(),
        })

        depth = 0
        # pareto optimal pruning search
        while True:
            depth += 1
            if depth > MAXTPATHDEPTH:
                break
            if depth < PRUNINGDEPTH:
                pruning_flag = False
            else:
                pruning_flag = True
            curlayer_nodenum = 0
            for idx, leaf in enumerate(tree.leaves(nid='root')):
                adjacent_nodes = getAdjacentNodes4Path(tree, leaf.identifier)
                cur_leaf_nodes = []
                if tree.depth(node=leaf.identifier) < depth-1:
                    continue
                pre_colinfo = leaf.data['colinfo']
                pre_load = leaf.data['load']
                pre_tpath = leaf.data['tpath']
                if leaf.identifier == "root":
                    pre_t = ""
                    basicTl_sliceinitialidx = 0
                else:
                    pre_t = leaf.identifier.split(' - ')[-1].split('|')[2]
                    basicTl_sliceinitialidx = basicTl.index(pre_t)
                for t in basicTl[basicTl_sliceinitialidx:]:
                    if t in ['sum', 'sub', 'mul', 'div']:
                        # clusters matching
                        pre_cluster_i = int(leaf.identifier.split(' - ')[-1].split('|')[4]) if leaf.identifier != "root" else -1
                        for i, cluster in enumerate(num_clusters):
                            if t == pre_t and i <= pre_cluster_i:
                                continue
                            if t in ['sub', 'div', 'mul'] and len(cluster) != 2:
                                continue
                            if len(cluster) > 1:
                                new_colname = "%s:(%s)" % (t, ",".join(cluster))
                                cur_colinfo = deepcopy(pre_colinfo)
                                cur_colinfo['col_names'] = pd.Index(list(cur_colinfo['col_names'])+[new_colname])
                                cur_colinfo['num_col_names'] = pd.Index(list(cur_colinfo['num_col_names'])+[new_colname])
                                cur_colinfo['col_type'][new_colname] = {
                                    "type": "real",
                                    "domain": None,
                                    "max": None,
                                    "min": None,
                                    "iskey": False
                                }
                                cur_colinfo['col_names_simi']["vectors"].append(mean_w2v(cur_colinfo['col_names_simi']["vectors"],
                                                                                         cur_colinfo['col_names'],
                                                                                         cluster))
                                cur_colinfo['new_col'].append(new_colname)
                                dim_load = mean_distance(cur_colinfo["dist_mat"]["wasserstein"],
                                                         cur_colinfo['num_col_names'],
                                                         cluster)
                                sem_load = mean_distance(cur_colinfo["col_names_simi"]["cosine"],
                                                         cur_colinfo['num_col_names'],
                                                         cluster)
                                cur_load = (pre_load[0]+cal_load[t]*(len(cluster)-1), pre_load[1]+dim_load, pre_load[2]+sem_load)
                                cur_tpath = deepcopy(pre_tpath)
                                if hasRANK(cluster):
                                    if len(cur_tpath) == 0 or cur_tpath[0]["t"] != "rank":
                                        cur_tpath = tpath([rank_tp]) + cur_tpath
                                cur_tpath.append({
                                    "t": t,
                                    "i_type": "==",
                                    "i": cluster,
                                    "o_type": "append",
                                    "args": (),
                                    "kwargs": {"axis": 1},
                                    "index": pd.Index([new_colname])
                                })
                                cur_id = leaf.identifier+' - '+str(depth)+'|'+str(curlayer_nodenum)+'|'+t+'|CM|'+str(i)

                                if pruning_flag:
                                    # pruning the controlled TPs
                                    controlled_flag = False
                                    for nid in adjacent_nodes:
                                        if tree.depth(node=nid) < PRUNINGDEPTH:
                                            continue
                                        if Load(tree[nid].data['load']) <= Load(cur_load) and Load(tree[nid].data['load']) != Load(cur_load):
                                            controlled_flag = True
                                            break
                                    if controlled_flag:
                                        continue

                                    cur_leaf_nodes.append(Node(tag=cur_id, identifier=cur_id, data={
                                        'colinfo': cur_colinfo,
                                        'load': cur_load,
                                        'tpath': cur_tpath,
                                    }))
                                else:
                                    # create nodes directly
                                    tree.create_node(tag=cur_id, identifier=cur_id, parent=leaf.identifier, data={
                                        'colinfo': cur_colinfo,
                                        'load': cur_load,
                                        'tpath': cur_tpath,
                                    })

                                curlayer_nodenum += 1

                    elif t == "rank":
                        pass
                    elif t == "aggr":
                        pass
                # prune controlled new nodes of current leaf and add remaining nodes
                if pruning_flag:
                    remaining_nodes = pruneControlledNodes(cur_leaf_nodes)
                    for node in remaining_nodes:
                        tree.add_node(node, parent=leaf.identifier)

        # search ending, have constructed the search tree
        print("in T", tname, ": search ending")
        for n in tree.all_nodes_itr():
            if n.identifier == "root":
                continue
            cur_tpath = n.data["tpath"]
            if cur_tpath[0]["t"] == "rank":
                rank_flag = True
            else:
                rank_flag = False
            likel = []
            equall = []
            for t in cur_tpath:
                if t["t"] == "rank":
                    continue
                if t["i_type"] == "like":
                    for like in t["i"]:
                        if like not in likel:
                            likel.append(like)
                if t["i_type"] == "==":
                    for equal in t["i"]:
                        if equal not in equall:
                            equall.append(equal)
                if isinstance(t["index"], pd.Index):
                    for new_colname in list(t["index"]):
                        if new_colname not in equall:
                            equall.append(new_colname)
            cur_tpath.append({
                "t": "select",
                "i_type": "like" if len(likel) > 0 else "==",
                "i": likel if len(likel) > 0 else equall,
                "o_type": "new_table",
                "args": (),
                "kwargs": {},
                "index": "default"
            })
            pool.append((0, cur_tpath))
        print("in T", tname, "TP pool size:", len(pool))
    elif isinstance(tinput, list):
        pass

    if MULTIPROCESS:
        while True:
            updatequeue()
    if not MULTIPROCESS:
        return pool

RANK_prefix = "RANKED "

def ranking(colinfo):
    """
    optional adding RANK columns for real number columns
    this function may process some unbalanced conditions in dimension distributions
    """
    new_colnames = [RANK_prefix+colname for colname in list(colinfo['num_col_names']) if colinfo["col_type"][colname]["type"] == "real"]
    if len(new_colnames) == 0:
        return colinfo, None
    cur_colinfo = deepcopy(colinfo)
    cur_colinfo['col_names'] = pd.Index(list(colinfo['col_names']) + new_colnames)
    cur_colinfo['num_col_names'] = pd.Index(list(colinfo['num_col_names']) + new_colnames)
    for new_colname in new_colnames:
        old_colname = new_colname[len(RANK_prefix):]
        old_col_type = colinfo['col_type'][old_colname]
        cur_col_type = {
            "type": "int",
            "domain": None,
            "max": None,
            "min": None,
            "iskey": False
        }
        cur_colinfo["col_type"][new_colname] = cur_col_type
        cur_colinfo['col_names_simi']["vectors"].append(colinfo['col_names_simi']["vectors"][list(colinfo["col_names"]).index(old_colname)])

    # add dim matching clusters
    for cluster in colinfo["dim_match"]["clusters"]:
        cur_colinfo["dim_match"]["clusters"].append([RANK_prefix+colname for colname in cluster])

    # add sem matching clusters
    for cluster in colinfo["col_names_simi"]["clusters"]:
        cur_colinfo["col_names_simi"]["clusters"].append([RANK_prefix+colname for colname in cluster])

    # generate optional RANK TP node
    TPnode = {
        "t": "rank",
        "i_type": "like",
        "i": ["float"],
        "o_type": "append",
        "args": (),
        "kwargs": {
            "axis": 0,
            "method": "first",
            "numeric_only": True,
            "na_option": "keep",
            "ascending": True,
            "pct": False,
        },
        "index": pd.Index(new_colnames)
    }

    return cur_colinfo, TPnode

def hasRANK(l):
    if isinstance(l, list):
        for s in l:
            if s.startswith(RANK_prefix):
                return True
    elif isinstance(l, str):
        if l.startswith(RANK_prefix):
            return True
    return False

def getAdjacentNodes4Path(tree, pid):
    ret = []
    ids = pid.split(' - ')
    for i in range(len(ids)-1):
        cpid = ' - '.join(ids[:i+1])
        ccid = ' - '.join(ids[:i+2])
        for c in tree.children(cpid):
            if c.identifier == ccid:
                continue
            ret.append(c.identifier)
    return ret

def pruneControlledNodes(nodes):
    ret = []
    for ni in nodes:
        controlled_flag = False
        for nj in nodes:
            if ni.identifier == nj.identifier:
                continue
            if Load(nj.data['load']) <= Load(ni.data['load']) and Load(nj.data['load']) != Load(ni.data['load']):
                controlled_flag = True
                break
        if not controlled_flag:
            ret.append(ni)
    return ret

def mean_w2v(vectors, allnames, names):
    if len(names) <= 1:
        print("error in <func mean_distance>, len(names) ==", len(names))
    if not isinstance(allnames, list):
        allnames = list(allnames)
    ret = None
    for i, name in enumerate(names):
        if name.startswith(RANK_prefix):
            name = name[len(RANK_prefix):]
        if i == 0:
            ret = deepcopy(vectors[allnames.index(name)])
        else:
            ret += vectors[allnames.index(name)]
    return ret / len(names)


def mean_distance(dist_mat, allnames, names):
    if len(names) <= 1:
        print("error in <func mean_distance>, len(names) ==", len(names))
    if not isinstance(allnames, list):
        allnames = list(allnames)
    if isinstance(dist_mat, pd.DataFrame):
        dist_mat = dist_mat.values
    ret = 0
    count = 0
    for i in range(len(names) - 1):
        n1 = names[i]
        if n1.startswith(RANK_prefix):
            n1 = n1[len(RANK_prefix):]
        idx1 = allnames.index(n1)
        for j in range(i + 1, len(names)):
            n2 = names[j]
            if n2.startswith(RANK_prefix):
                n2 = n2[len(RANK_prefix):]
            idx2 = allnames.index(n2)
            ret += dist_mat[idx1][idx2]
            count += 1
    return ret / count

