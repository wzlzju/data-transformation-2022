import os, sys, time
from copy import deepcopy
import pickle
import numpy as np
import pandas as pd

from treelib import Tree, Node

from spreadsheet import spreadsheet
from V import *
from T import *
from Tfunctions import *
from config import *
from utils import *
import score



def transform(data, coret=None, tpath=None):
    ndata = data
    transformation_functions = {
        "pca": Tpca,
        "tsne": Ttsne,
        "mds": Tmds,
        "umap": Tumap,
        "lda": Tlda,
        "dbscan": Tdbscan,
        "kmeans": Tkmeans,

        "null_nom1": Tnull,
        "null_num1": Tnull,

        "test": Ttest
    }

    # process the basic transformation path
    if tpath not in [None, "", " ", [], {}]:
        for t in tpath:
            # process input
            if t["i_type"] == "like":
                idata = ndata.select_dtypes(include=t["i"])
            elif t["i_type"] == "==":
                idata = ndata[t["i"]]
            else:
                print("error: unexpected basic T input type:", t["i_type"], "in <func transform>")
                raise Exception("error unexcepted T input type")

            # process T
            if t["t"] == "astype":
                odata = idata.astype(*t["args"], **t["kwargs"])
            elif t["t"] == "sum":
                odata = idata.apply(lambda x: x.sum(), *t["args"], **t["kwargs"])
            elif t["t"] == "mul":
                odata = idata.apply(lambda x: x.product(), *t["args"], **t["kwargs"])
            elif t["t"] == "select":
                odata = idata
            elif t["t"] == "rank":
                odata = idata.rank(*t["args"], **t["kwargs"]).astype("int64")
            elif t["t"] == "nominalize":
                if isinstance(idata, pd.Series):
                    idata = pd.DataFrame(idata)
                categoriesset = np.unique(idata.values)
                odata = idata.apply(lambda x: int(np.argwhere(categoriesset == x.values)), *t["args"], **t["kwargs"])
                odata = pd.concat([idata, odata], axis=1)
            else:
                print("error: unexpected basic T:", t["t"], "in <func transform>")
                raise Exception("error unexcepted T")

            # process index
            if isinstance(t["index"], str) and t["index"] == "default":
                pass
            else:
                if isinstance(odata, pd.Series):
                    odata = pd.DataFrame(odata)
                odata.columns = t["index"]

            # process output
            if t["o_type"] == "new_table":
                ndata = odata
            elif t["o_type"] == "append":
                ndata = pd.concat([ndata, odata], axis=1)
            elif t["o_type"] == "replace":
                ndata.drop(idata, axis=1)
                ndata = pd.concat([ndata, odata], axis=1)
            else:
                print("error: unexpected basic T output type:", t["o_type"], "in <func transform>")
                raise Exception("error unexcepted T output type")

            # convert ndata into DataFrame
            # guarantee the ndata is DataFrame before core T
            if isinstance(ndata, pd.Series):
                ndata = pd.DataFrame(ndata)


    # process the core transformation
    if coret not in [None, "", " ", [], {}]:
        ndata = transformation_functions[coret["name"]](data=ndata, para=coret["para"])

    return ndata


def Tpca(data, para):
    ndata = data.select_dtypes(include=["int", "float"])
    res = ppca(ndata, **para)   # numpy result

    return pd.DataFrame(res)

def Ttsne(data, para):
    ndata = data.select_dtypes(include=["int", "float"])
    res = ptsne(ndata, **para)  # numpy result

    return pd.DataFrame(res).astype("float64")

def Tmds(data, para):
    ndata = data.select_dtypes(include=["int", "float"])
    res = pmds(ndata, **para)  # numpy result

    return pd.DataFrame(res).astype("float64")

def Tumap(data, para):
    ndata = data.select_dtypes(include=["int", "float"])
    res = pumap(ndata, **para)  # numpy result

    return pd.DataFrame(res).astype("float64")

def Tlda(data, para):
    ndata = data.select_dtypes(include=["int", "float"])
    res = plda(ndata, **para)   # numpy result

    return pd.Series(res)

def Tdbscan(data, para):
    ndata = data.select_dtypes(include=["int", "float"])
    res = pdbscan(ndata, **para)   # numpy result

    return pd.Series(res)

def Tkmeans(data, para):
    ndata = data.select_dtypes(include=["int", "float"])
    res = pkmeans(ndata, **para)   # numpy result

    return pd.Series(res).astype("int64")

def Tnull(data, para):
    ndata = data.select_dtypes(include=["int", "float"])
    return ndata

def Ttest(data, para):
    return data



def printTP(tp, TAB=""):
    print(TAB, "Tpath:")
    for t in tp:
        print(TAB, "\t", t["t"])
        print(TAB, "\t\t", "i_type:", t["i_type"])
        print(TAB, "\t\t", "i:", t["i"])
        print(TAB, "\t\t", "o_type:", t["o_type"])
        print(TAB, "\t\t", "args:", t["args"])
        print(TAB, "\t\t", "kwargs:", t["kwargs"])
        print(TAB, "\t\t", "index:", str(t["index"]).replace("\n", ""))