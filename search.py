import json, csv
import os, sys, time
from copy import deepcopy
import pickle
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn

from treelib import Tree, Node
import matplotlib
from multiprocessing import Process, Pool, Queue
from _queue import Empty

from spreadsheet import spreadsheet
from Tengine import transform, printTP
from V import *
from T import *
from Tfunctions import *
from config import *
from searchthread import tpaththreadfunction
from utils import *
import score



class searchobj:
    def __init__(self, dataobj, search_bounds=MAXSEARCH, tpath_bounds=MAXTPATH, **kwargs):
        self.dataobj = dataobj
        self.search_bounds = search_bounds
        self.tpath_bounds = tpath_bounds
        self.processes = {}
        self.tpathpools = {}
        self.tpathsets = {}
        self.stree = None


    def presearch(self):
        for t in tlist.keys():
            if t not in numtl and t not in cattl:
                continue
            self.tpathsets[t] = set()
            rept = getRepT(t)
            if rept in self.tpathpools.keys():
                continue
            self.tpathpools[t] = Queue(1)
            print("start process:", t)
            self.processes[t] = Process(target=tpaththreadfunction, args=(t, self.dataobj.colinfo, self.tpathpools[t],))
            self.processes[t].start()
        time.sleep(1)
        print("in presearch:")
        for t in tlist.keys():
            if t not in numtl and t not in cattl:
                continue
            print("\t", t)
            ctpaths = self.tpathpools[getRepT(t)].get(True)
            for i, (v, ctpath) in enumerate(ctpaths):
                printTP(ctpath, TAB="\t\t")


    def postsearchinitialization(self):
        """
        post-search initialization
        construct the basic stree structure (vnode-inode-tnode), no tpnode
        :return: initial stree (without tpath nodes)
        """

        # construct search tree
        self.stree = Tree(deep=True)
        self.stree.create_node(tag='root', identifier='root')
        self.postsearchround = 0

        # traverse Vs
        for vidx, cvname in enumerate(list(vlist.keys())):
            cv = vlist[cvname]
            vid = "v-%d" % vidx
            vnode = Node(tag=vid, identifier=vid, data={
                "vname": cvname,
                "v": cv
            })
            self.stree.add_node(vnode, parent='root')

            # traverse input channels of V
            for iidx, cinputname in enumerate((cv["input"].keys())):
                cinput = cv["input"][cinputname]
                ctype = cinput["type"]
                if ctype == "num":
                    ctl = numtl
                elif ctype == "cat":
                    ctl = cattl
                else:
                    print("error: unexpected input type:", ctype, "of v:", cv)
                    raise Exception("error input type")

                iid = "i-%d-%d" % (vidx, iidx)
                inode = Node(tag=iid, identifier=iid, data={
                    "inputtype": ctype,
                    "inputname": cinputname,
                    "input": cinput
                })
                self.stree.add_node(inode, parent=vid)

                # traverse corresponding T list
                for tidx, ctname in enumerate(ctl):
                    ct = tlist[ctname]

                    # match the T output and V input
                    if cinput["dim"] is not None and ct["output"]["dim"] is not None and cinput["dim"] != ct["output"]["dim"]:
                        continue

                    tid = "t-%d-%d-%d" % (vidx, iidx, tidx)
                    tnode = Node(tag=tid, identifier=tid, data={
                        "tname": ctname,
                        "t": ct
                    })
                    self.stree.add_node(tnode, parent=iid)

        return self.stree

    def postsearch(self):
        """
        start a new round of post-search
        update self.stree according to the new discovered tpaths
        :return: the newest stree
        """
        try:
            self.postsearchround += 1
        except AttributeError as e:
            print("Don't forget to initialize the post-search via <class searchobj><func postsearchinitialization>. ")
            raise e

        # obtain current new tpaths for all core Ts
        ntpathsets = {}
        for t in tlist.keys():
            if t not in numtl and t not in cattl:
                continue
            ctpathset = set([pickle.dumps(ctpath) for cscore, ctpath in self.gettpath(t)])
            ntpathsets[t] = ctpathset - self.tpathsets[t]
            self.tpathsets[t] = self.tpathsets[t] | ntpathsets[t]

        # traverse Vs
        for vidx, cvname in enumerate(list(vlist.keys())):
            cv = vlist[cvname]

            # traverse input channels of V
            for iidx, cinputname in enumerate((cv["input"].keys())):
                cinput = cv["input"][cinputname]
                ctype = cinput["type"]
                if ctype == "num":
                    ctl = numtl
                elif ctype == "cat":
                    ctl = cattl
                else:
                    print("error: unexpected input type:", ctype, "of v:", cv)
                    raise Exception("error input type")

                # traverse corresponding T list
                for tidx, ctname in enumerate(ctl):
                    ct = tlist[ctname]

                    # match the T output and V input
                    if cinput["dim"] is not None and ct["output"]["dim"] is not None and cinput["dim"] != ct["output"][
                        "dim"]:
                        continue

                    tid = "t-%d-%d-%d" % (vidx, iidx, tidx)

                    # traverse basic transformation path of T (via self.gettpath())
                    ntpathset = ntpathsets[ctname]
                    for tpidx, sctpath in enumerate(ntpathset):
                        ctp = pickle.loads(sctpath)
                        tpid = "tp-%d-%d-%d-%d_%d" % (vidx, iidx, tidx, self.postsearchround, tpidx)
                        tnode = Node(tag=tpid, identifier=tpid, data={
                            "tp": ctp
                        })
                        self.stree.add_node(tnode, parent=tid)
        return self.stree

    def gettpath(self, t):
        if not isinstance(t, str):
            t = t["name"]
        pool = self.tpathpools[getRepT(t)].get(True)
        return pool
        # if t["name"] == "pca":
        #     return []
        # elif t["name"] == "lda":
        #     return []
        # else:
        #     print("warning: unexpected t:", t)
        #     return []


    def assemblevisdata(self, round=None):
        """
        assemble data for V
        it will return the newest assembled data and update self.visdata
        :param round: only tpnodes obtained in this round will be assemble
                        None: all rounds
                        <int>: a specific round
                        <list>: a list of specific rounds
        :return: the newest self.visdata
        """
        resdatabuffer = {}
        print("assembling ...")
        root = self.stree.get_node("root")
        vnl = self.stree.children("root")
        self.visdata = []
        for iter1, vn in enumerate(vnl):
            if DEBUG:
                print("vn:", vn)
            vd = {
                "_chart_type": vn.data["vname"]
            }
            inl = self.stree.children(vn.identifier)
            for iter2, in_ in enumerate(inl):
                if DEBUG:
                    print("\tin:", in_)
                if vd.get(in_.data["inputname"], None) is None:
                    vd[in_.data["inputname"]] = []
                paths = self.stree.subtree(in_.identifier).paths_to_leaves()
                for iter3, path in enumerate(paths):
                    if DEBUG:
                        print("\t\tpath:", path)
                    if len(path) == 1:
                        # no appropriate core T
                        # this time the paths list is like [['i-x-x']] and this incomplete path is the only element
                        continue
                    if len(path) == 2:
                        # no appropriate tpath
                        # this time the paths list is like [['i-x-x', 't-x-x-x']] and this incomplete path is the only element
                        continue
                    cpathr = int(path[-1].split("-")[-1].split("_")[0])
                    if tocontinue(cpathr, round, "round"):
                        continue
                    coret = self.stree.get_node(path[1]).data["t"]
                    tpath = self.stree.get_node(path[2]).data["tp"]
                    if DEBUG:
                        print("\t\t\tcore T:", coret)
                        printTP(tpath, TAB="\t\t\t")
                    separation = "<SEPARATION>".encode()
                    idxstr = pickle.dumps(coret) + separation + pickle.dumps(tpath)
                    ndata = resdatabuffer.get(idxstr, None)
                    if ndata is None:
                        ndata = transform(self.dataobj.data, coret, tpath)
                        resdatabuffer[idxstr] = ndata
                    vd[in_.data["inputname"]].append({
                        "data": ndata,
                        "coret": coret,
                        "tpath": tpath
                    })
            self.visdata.append(vd)

        return self.visdata

    def showtest(self, idx=None):
        """
        it will show a experimental vis of self.visdata in python environment
        :param: idx: only show V in idx
                        None: show all
        :return: None
        """
        matplotlib.style.use("seaborn")
        for _, cvisd in enumerate(self.visdata):
            if cvisd["_chart_type"].endswith("scatter"):
                xys = cvisd["xy"]
                colors = cvisd["color"]

                for ii, xy_obj in enumerate(xys):
                    xy = xy_obj["data"]
                    xy_coret = xy_obj["coret"]
                    xy_tpath = xy_obj["tpath"]
                    if tocontinue(ii, idx, "xy"):
                        continue

                    if isinstance(xy, pd.DataFrame):
                        pass
                    else:
                        print("error: unexpected xy format. excepted pandas.DataFrame, but got",
                              type(xy), "in <class searchobj><func showtest><branch scatter>")
                        raise Exception("error unexpected color format")

                    if DEBUG:
                        print("xy:")
                        print("core T:", xy_coret)
                        printTP(xy_tpath, TAB="")

                    for jj, color_obj in enumerate(colors):
                        color = color_obj["data"]
                        color_coret = color_obj["coret"]
                        color_tpath = color_obj["tpath"]
                        if tocontinue(jj, idx, "color"):
                            continue

                        # optional: only syhthesis V whose two input channels have same final-selected-tpnode
                        if ONLYVISUALIZESELECTIONMATCHINGCHANNELS and xy_coret["name"] in alignTl and color_coret["name"] in alignTl:
                            if xy_tpath[-1]["i_type"] != color_tpath[-1]["i_type"] or xy_tpath[-1]["i"] != color_tpath[-1]["i"]:
                                continue

                        if DEBUG:
                            print("\tcolor:")
                            print("\tcore T:", color_coret)
                            printTP(color_tpath, TAB="\t")

                        d = xy.values
                        d = d.T

                        x = d[0]
                        y = d[1]

                        # process the two color mode
                        if isinstance(color, pd.DataFrame):
                            color = color[color.columns[0]]
                        elif isinstance(color, pd.Series):
                            pass
                        else:
                            print("error: unexpected color format. excepted pandas.Series or pandas.DataFrame, but got",
                                  type(color), "in <class searchobj><func showtest><branch scatter>")
                            raise Exception("error unexpected color format")

                        if cvisd["_chart_type"] == "cat_scatter":
                            if color.dtype == "float64":
                                color = color.astype("int64")
                            if DEBUG:
                                if len(np.unique(color[color >= 0].values)) > 1:
                                    print("\tCDM:", score.CDM(xy.values, color.values))
                                elif len(np.unique(color[color >= 0].values)) == 1:
                                    pass
                                else:
                                    # all data are outliers, may be processed like above (==1)
                                    pass
                        elif cvisd["_chart_type"] == "num_scatter":
                            if color.dtype == "int64":
                                color = color.astype("float64")

                        # color data -> color  --from palette
                        if str(color.dtype).startswith("int"):
                            # nominal data -> color
                            # palette = [[141, 211, 199], [255, 255, 179], [190, 186, 218], [251, 128, 114], [128, 177, 211],
                            #            [253, 180, 98], [179, 222, 105], [252, 205, 229], [217, 217, 217], [188, 128, 189]]
                            # palette = [[v / 255 for v in c] for c in palette]
                            palette = seaborn.color_palette("muted", n_colors=max(color)+1)
                            # prepare for the outliers
                            palette.append(OUTLIERCOLOR)

                            c = np.array([palette[int(ci) % len(palette)] for ci in np.array(color)])
                        elif str(color.dtype).startswith("float"):
                            # numerical data -> color
                            color = color-min(color)
                            color = color/max(color)
                            palette = [[8, 48, 107], [222, 235, 247]]
                            palette = [[v / 255 for v in c] for c in palette]
                            palette = np.array(palette)
                            c = np.array([(palette[0]-palette[1])*float(ci) + palette[1] for ci in np.array(color)])

                        norm = plt.Normalize(1, len(set(color)))
                        cmap = plt.cm.RdYlGn

                        fig, ax = plt.subplots()
                        sc = plt.scatter(x, y, c=c, s=12, cmap=cmap, norm=norm)

                        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                            bbox=dict(boxstyle="round", fc="w"),
                                            arrowprops=dict(arrowstyle="->"))
                        annot.set_visible(False)

                        def update_annot(ind):
                            # print(ind)
                            pos = sc.get_offsets()[ind["ind"][0]]
                            annot.xy = pos
                            text = "\n".join(
                                [str((self.dataobj.data[self.dataobj.key if self.dataobj.key else self.dataobj.columnnames[0]])[int(cc)]) for cc in
                                 list(map(str, ind["ind"]))])
                            annot.set_text(text)
                            # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
                            # annot.get_bbox_patch().set_alpha(0.4)

                        def hover(event):
                            vis = annot.get_visible()
                            if event.inaxes == ax:
                                cont, ind = sc.contains(event)
                                if cont:
                                    update_annot(ind)
                                    annot.set_visible(True)
                                    fig.canvas.draw_idle()
                                else:
                                    if vis:
                                        annot.set_visible(False)
                                        fig.canvas.draw_idle()

                        fig.canvas.mpl_connect("motion_notify_event", hover)

                        plt.show()


            elif self.visdata["_chart_type"] == "bar":
                print("waiting for implementing")
            elif self.visdata["_chart_type"] == "line":
                print("waiting for implementing")
            else:
                print("error: unexpected vis chart type")
                raise Exception("error chat type")

    def destruct(self):
        for t in self.processes.keys():
            self.processes[t].terminate()




if __name__ == "__main__":
    sheet = spreadsheet("./testdata/LS1/WineQT.csv", encoding="unicode_escape", keep_default_na=False)
    #sheet = spreadsheet("./testdata/NetflixOriginals.csv", encoding="unicode_escape", keep_default_na=False)
    print(sheet.data)

    so = searchobj(dataobj=sheet)
    so.presearch()
    so.postsearchinitialization()
    stree = so.postsearch()
    visdata = so.assemblevisdata(round=1)
    so.showtest()
    # so.showtest(idx={"xy": [0, 1, 2], "color": [0, 1]})
    stree = so.postsearch()
    visdata = so.assemblevisdata(round=2)
    so.showtest()
    so.destruct()

