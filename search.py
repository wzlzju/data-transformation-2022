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
        self.tpathtree = None


    def presearch(self):
        for t in tlist.keys():
            if t not in numtl and t not in cattl:
                continue
            self.tpathsets[t] = set()
            rept = getRepT(t)
            if rept in self.tpathpools.keys():
                continue
            if MULTIPROCESS:
                self.tpathpools[t] = Queue(1)
                print("start process:", t)
                self.processes[t] = Process(target=tpaththreadfunction, args=(t, self.dataobj.colinfo, self.tpathpools[t],))
                self.processes[t].start()
            else:
                self.tpathpools[t] = tpaththreadfunction(t, self.dataobj.colinfo, None)
        if MULTIPROCESS:
            time.sleep(1)
        print("in presearch:")
        for t in tlist.keys():
            if t not in numtl and t not in cattl:
                continue
            print("\t", t)
            if MULTIPROCESS:
                ctpaths = self.tpathpools[getRepT(t)].get(True)
            else:
                ctpaths = self.tpathpools[getRepT(t)]
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
                    # if cinput["dim"] is not None and ct["output"]["dim"] is not None and cinput["dim"] != ct["output"]["dim"]:
                    #     continue
                    if cinput["dim"] != ct["output"]["dim"]:
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
            if ONLYNEWTPATH:
                ntpathsets[t] = ctpathset - self.tpathsets[t]
            else:
                ntpathsets[t] = ctpathset
            self.tpathsets[t] = self.tpathsets[t] | ntpathsets[t]

        # get basic T path tree for frontend
        self.tpathtree = Tree()
        self.tpathtree.create_node(tag="r", identifier="r", data=None)
        for ctname in numtl+cattl:
            ntpathset = ntpathsets[ctname]
            for sctpath in ntpathset:
                ctpath = pickle.loads(sctpath)
                pid = "r"
                cid = "r"
                for t in ctpath:
                    cid = pid + SEPERATION + str(t)
                    if cid not in self.tpathtree.is_branch(pid):
                        self.tpathtree.create_node(tag=cid, identifier=cid, parent=pid, data=None)
                    pid = cid
                cid = pid + SEPERATION + str(tlist[ctname])
                if cid not in self.tpathtree.is_branch(pid):
                    self.tpathtree.create_node(tag=cid, identifier=cid, parent=pid, data=None)



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
                    # if cinput["dim"] is not None and ct["output"]["dim"] is not None and cinput["dim"] != ct["output"]["dim"]:
                    #     continue
                    if cinput["dim"] != ct["output"]["dim"]:
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
        if MULTIPROCESS:
            pool = self.tpathpools[getRepT(t)].get(True)
        else:
            pool = self.tpathpools[getRepT(t)]
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
                        ndata, self.tpathtree = transform(self.dataobj.data, coret, tpath, self.tpathtree)
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
                            cat2legend = {}
                            if color.dtype == "int64":
                                for co in np.unique(color.values):
                                    if co < 0:
                                        cat2legend[co] = "outliers"
                                    else:
                                        cat2legend[co] = "cluster "+str(co+1)
                            elif color.dtype == "object":
                                idata = color
                                if isinstance(idata, pd.Series):
                                    idata = pd.DataFrame(idata)
                                categoriesset = np.unique(idata.values)
                                cat2legend = {idx: lege for idx, lege in enumerate(categoriesset)}
                                odata = idata.apply(lambda x: int(np.argwhere(categoriesset == x.values)), axis=1)
                                color = odata
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

                        # g = score.dotGraph(xy.values)
                        # g.minSpanTree()
                        # print(g.outlying_value())
                        # print(g.skew_value())
                        # print(g.striated_value())
                        # print(g.stringy_value())
                        # print(g.straight_value())
                        # print(g.spearman_value())
                        # print(g.clumpy_value())

                        # color data -> color  --from palette
                        legend2color = {}
                        if str(color.dtype).startswith("int"):
                            # nominal data -> color
                            # palette = [[141, 211, 199], [255, 255, 179], [190, 186, 218], [251, 128, 114], [128, 177, 211],
                            #            [253, 180, 98], [179, 222, 105], [252, 205, 229], [217, 217, 217], [188, 128, 189]]
                            # palette = [[v / 255 for v in c] for c in palette]
                            palette = seaborn.color_palette("muted", n_colors=max(color)+1)
                            # prepare for the outliers
                            palette.append(OUTLIERCOLOR)

                            c = np.array([palette[int(ci) % len(palette)] for ci in np.array(color)])
                            for cat in cat2legend.keys():
                                legend2color[cat2legend[cat]] = palette[int(cat) % len(palette)]
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

                        plt.xlabel(xy.columns[0])
                        plt.ylabel(xy.columns[1])
                        plt.show()


            elif cvisd["_chart_type"].endswith("bar"):
                xs = cvisd["x"]
                ys = cvisd["y"]

                for ii, x_obj in enumerate(xs):
                    x = x_obj["data"]
                    x_coret = x_obj["coret"]
                    x_tpath = x_obj["tpath"]
                    if tocontinue(ii, idx, "x"):
                        continue
                    if isinstance(x, pd.DataFrame):
                        pass
                    elif isinstance(x, pd.Series):
                        x = pd.DataFrame(x)
                    else:
                        print("error: unexpected y format. excepted pandas.DataFrame, but got",
                              type(x), "in <class searchobj><func showtest><branch line>")
                        raise Exception("error unexpected color format")

                    if DEBUG:
                        print("x:")
                        print("core T:", x_coret)
                        printTP(x_tpath, TAB="")

                    x.columns = pd.Index(["Category by "+x_coret["name"].upper()])

                    for jj, y_obj in enumerate(ys):
                        y = y_obj["data"]
                        y_coret = y_obj["coret"]
                        y_tpath = y_obj["tpath"]
                        if tocontinue(jj, idx, "y"):
                            continue
                        if isinstance(y, pd.DataFrame):
                            pass
                        elif isinstance(y, pd.Series):
                            y = pd.DataFrame(y)
                        else:
                            print("error: unexpected y format. excepted pandas.DataFrame, but got",
                                  type(y), "in <class searchobj><func showtest><branch line>")
                            raise Exception("error unexpected color format")

                        if DEBUG:
                            print("y:")
                            print("core T:", y_coret)
                            printTP(y_tpath, TAB="")

                        # reduce y
                        if len(y.columns) > MAXBARNUMINCHART:
                            tarcol = list(y.columns)[:MAXBARNUMINCHART]
                            for col in list(y.columns)[MAXBARNUMINCHART:]:
                                if col not in self.dataobj.data.columns:
                                    tarcol.append(col)
                            y = y[tarcol]

                        tmpxy = pd.concat([x, y], axis=1)
                        groups = tmpxy.groupby(x.columns[0])
                        y = None
                        for xcat, gy in groups:
                            if cvisd["_chart_type"].startswith("sum"):
                                gy = gy.select_dtypes(include=["int", "float"])
                                gya = gy.agg(sum)
                                gya[x.columns[0]] = xcat
                                ny = pd.DataFrame(gya.values.reshape(1, len(gya)), columns=pd.Index(["SUM(%s)" % i for i in gya.index]))
                                if y is None:
                                    y = ny
                                else:
                                    y = pd.concat([y, ny])
                            elif cvisd["_chart_type"].startswith("count"):
                                ny = pd.DataFrame([[len(gy), xcat]], columns=pd.Index(["COUNT", x.columns[0]]))
                                if y is None:
                                    y = ny
                                else:
                                    y = pd.concat([y, ny])
                        ndata = y

                        def create_multi_bars(labels, datas, xlabel="", ylabel="", legend=None, tick_step=1, group_gap=0.2, bar_gap=0):
                            ticks = np.arange(len(labels)) * tick_step
                            group_num = len(datas)
                            group_width = tick_step - group_gap
                            bar_span = group_width / group_num
                            bar_width = bar_span - bar_gap
                            baseline_x = ticks - (group_width - bar_span) / 2
                            for index, y in enumerate(datas):
                                plt.bar(baseline_x + index * bar_span, y, bar_width, label= legend[index] if legend is not None else str(index))
                            plt.xlabel(xlabel)
                            plt.ylabel(ylabel)
                            plt.xticks(ticks, labels)
                            plt.legend()
                            plt.show()
                        create_multi_bars(ndata[ndata.columns[-1]].values,
                                          ndata[ndata.columns[:-1]].values.T,
                                          xlabel=ndata.columns[-1],
                                          ylabel="",
                                          legend=list(ndata.columns[:-1]))

            elif cvisd["_chart_type"].endswith("line"):
                if cvisd["_chart_type"].startswith("ord"):
                    ys = cvisd["y"]

                    for ii, y_obj in enumerate(ys):
                        y = y_obj["data"]
                        y_coret = y_obj["coret"]
                        y_tpath = y_obj["tpath"]
                        if tocontinue(ii, idx, "y"):
                            continue
                        if isinstance(y, pd.DataFrame):
                            pass
                        elif isinstance(y, pd.Series):
                            y = pd.DataFrame(y)
                        else:
                            print("error: unexpected y format. excepted pandas.DataFrame, but got",
                                  type(y), "in <class searchobj><func showtest><branch line>")
                            raise Exception("error unexpected color format")

                        if DEBUG:
                            print("y:")
                            print("core T:", y_coret)
                            printTP(y_tpath, TAB="")

                        # reduce y
                        if len(y.columns) > MAXLINENUMINCHART:
                            tarcol = list(y.columns)[:MAXLINENUMINCHART]
                            for col in list(y.columns)[MAXLINENUMINCHART:]:
                                if col not in self.dataobj.data.columns:
                                    tarcol.append(col)
                            y = y[tarcol]

                        x = range(len(y))
                        plt.figure()
                        for col in y.columns:
                            plt.plot(x, y[col], label=col)
                            plt.legend()
                        if len(y.columns) == 1:
                            plt.ylabel(y.columns[0])
                        plt.show()
                elif cvisd["_chart_type"].startswith("rel"):
                    xs = cvisd["x"]
                    ys = cvisd["y"]

                    for ii, x_obj in enumerate(xs):
                        x = x_obj["data"]
                        x_coret = x_obj["coret"]
                        x_tpath = x_obj["tpath"]
                        if tocontinue(ii, idx, "x"):
                            continue
                        if isinstance(x, pd.DataFrame):
                            pass
                        elif isinstance(x, pd.Series):
                            x = pd.DataFrame(x)
                        else:
                            print("error: unexpected y format. excepted pandas.DataFrame, but got",
                                  type(x), "in <class searchobj><func showtest><branch line>")
                            raise Exception("error unexpected color format")

                        if DEBUG:
                            print("x:")
                            print("core T:", x_coret)
                            printTP(x_tpath, TAB="")

                        sortTOKEN = "<SORTBY>"
                        x.columns = pd.Index([sortTOKEN + x.columns[0]])

                        for jj, y_obj in enumerate(ys):
                            y = y_obj["data"]
                            y_coret = y_obj["coret"]
                            y_tpath = y_obj["tpath"]
                            if tocontinue(jj, idx, "y"):
                                continue
                            if isinstance(y, pd.DataFrame):
                                pass
                            elif isinstance(y, pd.Series):
                                y = pd.DataFrame(y)
                            else:
                                print("error: unexpected y format. excepted pandas.DataFrame, but got",
                                      type(y), "in <class searchobj><func showtest><branch line>")
                                raise Exception("error unexpected color format")

                            if DEBUG:
                                print("y:")
                                print("core T:", y_coret)
                                printTP(y_tpath, TAB="")

                            # reduce y
                            if len(y.columns) > MAXLINENUMINCHART:
                                tarcol = list(y.columns)[:MAXLINENUMINCHART]
                                for col in list(y.columns)[MAXLINENUMINCHART:]:
                                    if col not in self.dataobj.data.columns:
                                        tarcol.append(col)
                                y = y[tarcol]

                            tmpxy = pd.concat([x, y], axis=1)
                            tmpxy = tmpxy.sort_values(by=tmpxy.columns[0])
                            tmpx = tmpxy[tmpxy.columns[0]]
                            tmpy = tmpxy[tmpxy.columns[1:]]
                            plt.figure()
                            for col in tmpy.columns:
                                plt.plot(tmpx, tmpy[col], label=col)
                                plt.legend()
                            plt.xlabel(tmpxy.columns[0].replace(sortTOKEN, ""))
                            if len(y.columns) == 1:
                                plt.ylabel(y.columns[0])
                            plt.show()
            else:
                print("error: unexpected vis chart type")
                raise Exception("error chat type")

    def assembleandevaluevis(self, idx=None):
        """
        assemble V using self.visdata, save results in self.vis
        :param: idx: only show V in idx
                        None: show all
        """
        self.visbuffer = {
            "scatter": [],
            "line": [],
            "cat_line": [],
            "bar": []
        }
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
                        raise Exception("error unexpected xy format")

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

                        cs = {}
                        cat2legend = {}
                        if cvisd["_chart_type"] == "cat_scatter":
                            if color.dtype == "float64":
                                color = color.astype("int64")
                            if color.dtype == "int64":
                                for co in np.unique(color.values):
                                    if co < 0:
                                        cat2legend[co] = "outliers"
                                    else:
                                        cat2legend[co] = "cluster " + str(co + 1)
                            elif color.dtype == "object":
                                idata = color
                                if isinstance(idata, pd.Series):
                                    idata = pd.DataFrame(idata)
                                categoriesset = np.unique(idata.values)
                                cat2legend = {idx: lege for idx, lege in enumerate(categoriesset)}
                                odata = idata.apply(lambda x: int(np.argwhere(categoriesset == x.values)), axis=1)
                                color = odata
                            if DEBUG:
                                if len(np.unique(color[color >= 0].values)) > 1:
                                    cs["CDM"] = score.CDM(xy.values, color.values)
                                elif len(np.unique(color[color >= 0].values)) == 1:
                                    cs["CDM"] = 0
                                else:
                                    # all data are outliers, may be processed like above (==1)
                                    cs["CDM"] = 0
                        elif cvisd["_chart_type"] == "num_scatter":
                            if color.dtype == "int64":
                                color = color.astype("float64")
                            color = color - min(color)
                            color = color / max(color)
                            tmpcolor = color.apply(lambda x: int(x*4) if x < 1 else 3)
                            cs["CDM"] = score.CDM(xy.values, tmpcolor.values)

                        # color data -> color  --from palette
                        legend2color = {}
                        if str(color.dtype).startswith("int"):
                            # nominal data -> color
                            # palette = [[141, 211, 199], [255, 255, 179], [190, 186, 218], [251, 128, 114], [128, 177, 211],
                            #            [253, 180, 98], [179, 222, 105], [252, 205, 229], [217, 217, 217], [188, 128, 189]]
                            # palette = [[v / 255 for v in c] for c in palette]
                            palette = seaborn.color_palette("muted", n_colors=max(color)+1)
                            # prepare for the outliers
                            palette.append(OUTLIERCOLOR)

                            c = np.array([palette[int(ci) % len(palette)] for ci in np.array(color)])
                            for cat in cat2legend.keys():
                                legend2color[cat2legend[cat]] = palette[int(cat) % len(palette)]
                        elif str(color.dtype).startswith("float"):
                            # numerical data -> color
                            color = color-min(color)
                            color = color/max(color)
                            palette = [[8, 48, 107], [222, 235, 247]]
                            palette = [[v / 255 for v in c] for c in palette]
                            palette = np.array(palette)
                            c = np.array([(palette[0]-palette[1])*float(ci) + palette[1] for ci in np.array(color)])

                        self.visbuffer["scatter"].append((mean(cs.values()), {
                            "pnodes": {
                                "xy": "r"+SEPERATION+SEPERATION.join([str(t) for t in xy_tpath])+SEPERATION+str(xy_coret),
                                "color": "r"+SEPERATION+SEPERATION.join([str(t) for t in color_tpath])+SEPERATION+str(color_coret)
                            },
                            "chart_type": "scatter",
                            "data": [{
                                "x": float(x[i]),
                                "y": float(y[i]),
                                "color": [float(ti) for ti in c[i]],
                                "text": str((self.dataobj.data[self.dataobj.key if self.dataobj.key else self.dataobj.columnnames[0]])[i])
                            } for i in range(len(color))],
                            "legend": legend2color,
                            "xlabel": xy.columns[0],
                            "ylabel": xy.columns[1]
                        }))

            elif cvisd["_chart_type"].endswith("bar"):
                xs = cvisd["x"]
                ys = cvisd["y"]

                for ii, x_obj in enumerate(xs):
                    x = x_obj["data"]
                    x_coret = x_obj["coret"]
                    x_tpath = x_obj["tpath"]
                    if tocontinue(ii, idx, "x"):
                        continue
                    if isinstance(x, pd.DataFrame):
                        pass
                    elif isinstance(x, pd.Series):
                        x = pd.DataFrame(x)
                    else:
                        print("error: unexpected y format. excepted pandas.DataFrame, but got",
                              type(x), "in <class searchobj><func showtest><branch line>")
                        raise Exception("error unexpected color format")

                    if DEBUG:
                        print("x:")
                        print("core T:", x_coret)
                        printTP(x_tpath, TAB="")

                    x.columns = pd.Index(["Category by " + x_coret["name"].upper()])

                    for jj, y_obj in enumerate(ys):
                        y = y_obj["data"]
                        y_coret = y_obj["coret"]
                        y_tpath = y_obj["tpath"]
                        if tocontinue(jj, idx, "y"):
                            continue
                        if isinstance(y, pd.DataFrame):
                            pass
                        elif isinstance(y, pd.Series):
                            y = pd.DataFrame(y)
                        else:
                            print("error: unexpected y format. excepted pandas.DataFrame, but got",
                                  type(y), "in <class searchobj><func showtest><branch line>")
                            raise Exception("error unexpected color format")

                        if DEBUG:
                            print("y:")
                            print("core T:", y_coret)
                            printTP(y_tpath, TAB="")

                        # reduce y
                        if len(y.columns) > MAXBARNUMINCHART:
                            tarcol = list(y.columns)[:MAXBARNUMINCHART]
                            for col in list(y.columns)[MAXBARNUMINCHART:]:
                                if col not in self.dataobj.data.columns:
                                    tarcol.append(col)
                            y = y[tarcol]

                        tmpxy = pd.concat([x, y], axis=1)
                        groups = tmpxy.groupby(x.columns[0])
                        y = None
                        for xcat, gy in groups:
                            if cvisd["_chart_type"].startswith("sum"):
                                gy = gy.select_dtypes(include=["int", "float"])
                                gya = gy.agg(sum)
                                gya[x.columns[0]] = xcat
                                ny = pd.DataFrame(gya.values.reshape(1, len(gya)),
                                                  columns=pd.Index(["SUM(%s)" % i for i in gya.index]))
                                if y is None:
                                    y = ny
                                else:
                                    y = pd.concat([y, ny])
                            elif cvisd["_chart_type"].startswith("count"):
                                ny = pd.DataFrame([[len(gy), xcat]], columns=pd.Index(["COUNT", x.columns[0]]))
                                if y is None:
                                    y = ny
                                else:
                                    y = pd.concat([y, ny])
                        y.index = pd.RangeIndex(len(y))
                        ndata = y
                        y = y[y.columns[:-1]]
                        y = y.astype("float64")

                        cs = {}

                        cs["outno1"] = mean([score.significance_outstanding1(y[col].values) for col in y.columns])
                        cs["lincor"] = mean([score.significance_linearcorrelation(y[col].values) for col in y.columns])
                        if len(y.columns) >= 2:
                            corl = []
                            for i in range(len(y.columns) - 1):
                                for j in range(i + 1, len(y.columns)):
                                    corl.append(score.significance_correlation(np.array([y[y.columns[i]].values, y[y.columns[j]].values])))
                            cs["cor"] = mean(corl)

                        self.visbuffer["bar"].append((mean(cs.values()), {
                            "pnodes": {
                                "y": "r" + SEPERATION + SEPERATION.join([str(t) for t in y_tpath]) + SEPERATION + str(
                                    y_coret)
                            },
                            "chart_type": "bar",
                            "data": [{
                                "x": ndata[ndata.columns[-1]][i],
                                "y": [float(y[col][i]) for col in y.columns],
                                "text": ""
                            } for i in range(len(y))],
                            "legend": list(y.columns),
                            "xlabel": ndata.columns[-1],
                            "ylabel": ""
                        }))

            elif cvisd["_chart_type"].endswith("line"):
                if cvisd["_chart_type"].startswith("ord"):
                    ys = cvisd["y"]

                    for ii, y_obj in enumerate(ys):
                        y = y_obj["data"]
                        y_coret = y_obj["coret"]
                        y_tpath = y_obj["tpath"]
                        if tocontinue(ii, idx, "y"):
                            continue
                        if isinstance(y, pd.DataFrame):
                            pass
                        elif isinstance(y, pd.Series):
                            y = pd.DataFrame(y)
                        else:
                            print("error: unexpected y format. excepted pandas.DataFrame, but got",
                                  type(y), "in <class searchobj><func showtest><branch line>")
                            raise Exception("error unexpected color format")

                        if DEBUG:
                            print("y:")
                            print("core T:", y_coret)
                            printTP(y_tpath, TAB="")

                        # reduce y
                        if len(y.columns) > MAXLINENUMINCHART:
                            tarcol = list(y.columns)[:MAXLINENUMINCHART]
                            for col in list(y.columns)[MAXLINENUMINCHART:]:
                                if col not in self.dataobj.data.columns:
                                    tarcol.append(col)
                            y = y[tarcol]

                        catflag = False
                        yaxis = {}
                        if y.values.dtype == "object" and len(y.columns) == 1:
                            catflag = True
                            dataeleset = np.unique(y.values)
                            yaxis = {i+1: e for i, e in enumerate(dataeleset)}
                            data = np.array([int(np.argwhere(dataeleset == i)) + 1 for i in y.values])
                            y = pd.DataFrame(data, columns=y.columns)

                        cs = {}
                        x = list(range(len(y)))

                        cs["outno1"] = mean([score.significance_outstanding1(y[col].values) for col in y.columns])
                        cs["lincor"] = mean([score.significance_linearcorrelation(y[col].values) for col in y.columns])
                        if len(y.columns) >= 2:
                            corl = []
                            for i in range(len(y.columns) - 1):
                                for j in range(i + 1, len(y.columns)):
                                    corl.append(score.significance_correlation(np.array([y[y.columns[i]].values, y[y.columns[j]].values])))
                            cs["cor"] = mean(corl)

                        if not catflag:
                            self.visbuffer["line"].append((mean(cs.values()), {
                                "pnodes": {
                                    "y": "r" + SEPERATION + SEPERATION.join([str(t) for t in y_tpath]) + SEPERATION + str(y_coret)
                                },
                                "chart_type": "line",
                                "data": [{
                                    "x": x[i],
                                    "y": [float(y[col][i]) for col in y.columns],
                                    "text": str((self.dataobj.data[
                                        self.dataobj.key if self.dataobj.key else self.dataobj.columnnames[0]])[i])
                                } for i in range(len(y))],
                                "legend": list(y.columns),
                                "xlabel": "",
                                "ylabel": y.columns[0] if len(y.columns) == 1 else ""
                            }))
                        else:
                            self.visbuffer["cat_line"].append((mean(cs.values()), {
                                "pnodes": {
                                    "y": "r" + SEPERATION + SEPERATION.join(
                                        [str(t) for t in y_tpath]) + SEPERATION + str(y_coret)
                                },
                                "chart_type": "cat_line",
                                "data": [{
                                    "x": x[i],
                                    "y": [int(y[col][i]) for col in y.columns],
                                    "text": str((self.dataobj.data[
                                        self.dataobj.key if self.dataobj.key else self.dataobj.columnnames[0]])[i])
                                } for i in range(len(y))],
                                "legend": list(y.columns),
                                "xlabel": "",
                                "ylabel": y.columns[0] if len(y.columns) == 1 else "",
                                "yaxis": yaxis
                            }))

                elif cvisd["_chart_type"].startswith("rel"):
                    xs = cvisd["x"]
                    ys = cvisd["y"]

                    for ii, x_obj in enumerate(xs):
                        x = x_obj["data"]
                        x_coret = x_obj["coret"]
                        x_tpath = x_obj["tpath"]
                        if tocontinue(ii, idx, "x"):
                            continue
                        if isinstance(x, pd.DataFrame):
                            pass
                        elif isinstance(x, pd.Series):
                            x = pd.DataFrame(x)
                        else:
                            print("error: unexpected y format. excepted pandas.DataFrame, but got",
                                  type(x), "in <class searchobj><func showtest><branch line>")
                            raise Exception("error unexpected color format")

                        if DEBUG:
                            print("x:")
                            print("core T:", x_coret)
                            printTP(x_tpath, TAB="")

                        sortTOKEN = "<SORTBY>"
                        x.columns = pd.Index([sortTOKEN + x.columns[0]])

                        for jj, y_obj in enumerate(ys):
                            y = y_obj["data"]
                            y_coret = y_obj["coret"]
                            y_tpath = y_obj["tpath"]
                            if tocontinue(jj, idx, "y"):
                                continue
                            if isinstance(y, pd.DataFrame):
                                pass
                            elif isinstance(y, pd.Series):
                                y = pd.DataFrame(y)
                            else:
                                print("error: unexpected y format. excepted pandas.DataFrame, but got",
                                      type(y), "in <class searchobj><func showtest><branch line>")
                                raise Exception("error unexpected color format")

                            if DEBUG:
                                print("y:")
                                print("core T:", y_coret)
                                printTP(y_tpath, TAB="")

                            # reduce y
                            if len(y.columns) > MAXLINENUMINCHART:
                                tarcol = list(y.columns)[:MAXLINENUMINCHART]
                                for col in list(y.columns)[MAXLINENUMINCHART:]:
                                    if col not in self.dataobj.data.columns:
                                        tarcol.append(col)
                                y = y[tarcol]

                            catflag = False
                            yaxis = {}
                            if y.values.dtype == "object" and len(y.columns) == 1:
                                catflag = True
                                dataeleset = np.unique(y.values)
                                yaxis = {i + 1: e for i, e in enumerate(dataeleset)}
                                data = np.array([int(np.argwhere(dataeleset == i)) + 1 for i in y.values])
                                y = pd.DataFrame(data, columns=y.columns)

                            tmpxy = pd.concat([x, y], axis=1)
                            tmpxy = tmpxy.sort_values(by=tmpxy.columns[0])
                            xrank = x[x.columns[0]].rank()
                            tmpx = tmpxy[tmpxy.columns[0]]
                            tmpy = tmpxy[tmpxy.columns[1:]]

                            cs = {}

                            cs["outno1"] = mean([score.significance_outstanding1(tmpy[col].values) for col in tmpy.columns])
                            cs["lincor"] = mean([score.significance_linearcorrelation(tmpy[col].values) for col in tmpy.columns])
                            if len(y.columns) >= 2:
                                corl = []
                                for i in range(len(tmpy.columns) - 1):
                                    for j in range(i + 1, len(tmpy.columns)):
                                        corl.append(score.significance_correlation(
                                            np.array([tmpy[tmpy.columns[i]].values, tmpy[tmpy.columns[j]].values])))
                                cs["cor"] = mean(corl)

                            if not catflag:
                                self.visbuffer["line"].append((mean(cs.values()), {
                                    "pnodes": {
                                        "x": "r" + SEPERATION + SEPERATION.join(
                                            [str(t) for t in x_tpath]) + SEPERATION + str(x_coret),
                                        "y": "r" + SEPERATION + SEPERATION.join(
                                            [str(t) for t in y_tpath]) + SEPERATION + str(y_coret)
                                    },
                                    "chart_type": "line",
                                    "data": [{
                                        "x": tmpx[i],
                                        "y": [float(tmpy[col][i]) for col in tmpy.columns],
                                        "text": str((self.dataobj.data[
                                            self.dataobj.key if self.dataobj.key else self.dataobj.columnnames[0]])[int(np.argwhere(xrank.values==(i+1)))])
                                    } for i in range(len(tmpy))],
                                    "legend": list(tmpy.columns),
                                    "xlabel": tmpxy.columns[0].replace(sortTOKEN, ""),
                                    "ylabel": tmpy.columns[0] if len(tmpy.columns) == 1 else ""
                                }))
                            else:
                                self.visbuffer["cat_line"].append((mean(cs.values()), {
                                    "pnodes": {
                                        "x": "r" + SEPERATION + SEPERATION.join(
                                            [str(t) for t in x_tpath]) + SEPERATION + str(x_coret),
                                        "y": "r" + SEPERATION + SEPERATION.join(
                                            [str(t) for t in y_tpath]) + SEPERATION + str(y_coret)
                                    },
                                    "chart_type": "cat_line",
                                    "data": [{
                                        "x": tmpx[i],
                                        "y": [int(tmpy[col][i]) for col in tmpy.columns],
                                        "text": str((self.dataobj.data[
                                            self.dataobj.key if self.dataobj.key else self.dataobj.columnnames[0]])[
                                                        int(np.argwhere(xrank.values == (i + 1)))])
                                    } for i in range(len(tmpy))],
                                    "legend": list(tmpy.columns),
                                    "xlabel": tmpxy.columns[0].replace(sortTOKEN, ""),
                                    "ylabel": tmpy.columns[0] if len(tmpy.columns) == 1 else "",
                                    "yaxis": yaxis
                                }))

            else:
                print("error: unexpected vis chart type")
                raise Exception("error chat type")

        # evaluate
        self.visbuffer["scatter"].sort(key=lambda x: x[0], reverse=True)
        self.visbuffer["line"].sort(key=lambda x: x[0], reverse=True)
        self.visbuffer["cat_line"].sort(key=lambda x: x[0], reverse=True)
        self.visbuffer["bar"].sort(key=lambda x: x[0], reverse=True)
        self.vis = self.visbuffer["scatter"][:min(int(len(self.visbuffer["scatter"])*RECOMMENDPCT)+1, MAXSCATTER)] + \
                    self.visbuffer["line"][:min(int(len(self.visbuffer["line"]) * RECOMMENDPCT) + 1, MAXLINE)] + \
                    self.visbuffer["cat_line"][:min(int(len(self.visbuffer["cat_line"]) * RECOMMENDPCT) + 1, MAXCATLINE)] + \
                    self.visbuffer["bar"][:min(int(len(self.visbuffer["bar"]) * RECOMMENDPCT) + 1, MAXBAR)]

    def assembleTtree(self):
        """
        :return: {
            "nodes": [{
                "id": ,
                "node_type": ,  # "D"/"V"
                "data": {       # if "node_type" is "D"
                    "headers": [ ]
                }
                "data": {       # if "node_type" is "V"
                    "chart_type": "scatter",
                    "data": [{
                        "x": ,
                        "y": ,
                        "color":
                    }]
                }
            }],
            "edges": [{
                "from": ,   # id
                "to": ,     # id
                ("data":    # )
            }],
            "vis_list": [{
                "chart_type": "scatter",
                "data": [{
                    "x": ,
                    "y": ,
                    "color":
                }],
                "paths": {
                    "nodes": [ ],   # id
                    "edges": [{
                        "from": ,   # id
                        "to":       # id
                    }]
                }
            }]
        }
        """
        ret = {
            "nodes": [],
            "edges": [],
            "vis_list": []
        }
        node_ids = ["r"]
        ret["nodes"].append({
            "id": "r",
            "node_type": "D",
            "data": {
                "headers": self.tpathtree["r"].data
            }
        })
        for vnode in self.vis:
            vpnodes = vnode[1]["pnodes"]
            vchart_type = vnode[1]["chart_type"]
            vdata = vnode[1]["data"]
            vlegend = vnode[1]["legend"]
            vxlabel = vnode[1]["xlabel"]
            vylabel = vnode[1]["ylabel"]
            vyaxis =  vnode[1].get("yaxis", {})
            vid = vchart_type + "<VIS>" + (SEPERATION + SEPERATION).join(vpnodes.values())
            ret["nodes"].append({
                "id": vid,
                "node_type": "V",
                "data": {
                    "chart_type": vchart_type,
                    "data": vdata,
                    "legend": vlegend,
                    "xlabel": vxlabel,
                    "ylabel": vylabel,
                    "yaxis": vyaxis
                }
            })
            ret["vis_list"].append({
                "chart_type": vchart_type,
                "data": vdata,
                "legend": vlegend,
                "xlabel": vxlabel,
                "ylabel": vylabel,
                "yaxis": vyaxis,
                "paths": {
                    "nodes": ["r", vid],
                    "edges": []
                }
            })
            for vpnode_channel in vpnodes.keys():
                vpnode = vpnodes[vpnode_channel]
                ts = vpnode.split(SEPERATION)
                cid = "r"
                for i in range(0, len(ts)-1):
                    pid = SEPERATION.join(ts[:i+1])
                    cid = pid + SEPERATION + ts[i+1]
                    if cid not in node_ids:
                        node_ids.append(cid)
                        ret["nodes"].append({
                            "id": cid,
                            "node_type": "D",
                            "data": {
                                "headers": self.tpathtree[cid].data
                            }
                        })
                        ret["edges"].append({
                            "from": pid,
                            "to": cid
                        })
                    if cid not in ret["vis_list"][-1]["paths"]["nodes"]:
                        ret["vis_list"][-1]["paths"]["nodes"].append(cid)
                        ret["vis_list"][-1]["paths"]["edges"].append({
                            "from": pid,
                            "to": cid
                        })
                ret["edges"].append({
                    "from": cid,
                    "to": vid,
                    "data": None
                })
                ret["vis_list"][-1]["paths"]["edges"].append({
                    "from": cid,
                    "to": vid
                })
        return ret


    def deconstruct(self):
        if MULTIPROCESS:
            for t in self.processes.keys():
                self.processes[t].terminate()




if __name__ == "__main__":
    sheet = spreadsheet("./testdata/ie19b.csv", encoding="unicode_escape", keep_default_na=False)
    # sheet = spreadsheet("./testdata/training1.csv", encoding="unicode_escape", keep_default_na=False)
    #sheet = spreadsheet("./testdata/ZYF1/req0215/iris.csv", encoding="unicode_escape", keep_default_na=False)
    #sheet = spreadsheet("./testdata/NetflixOriginals.csv", encoding="unicode_escape", keep_default_na=False)
    print(sheet.data)

    so = searchobj(dataobj=sheet)
    so.presearch()
    so.postsearchinitialization()
    stree = so.postsearch()
    visdata = so.assemblevisdata(round=1)
    # so.showtest()
    # so.showtest(idx={"xy": [0, 1, 2], "color": [0, 1]})
    so.assembleandevaluevis()
    tree2front = so.assembleTtree()
    print(tree2front)


