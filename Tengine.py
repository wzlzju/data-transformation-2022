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



def transform(data, coret=None, tpath=None, tpathtree=None):
    ndata = data
    if tpathtree is not None:
        if tpathtree["r"].data is None:
            tpathtree["r"].data = list(ndata.columns)
    transformation_functions = {
        "pca": Tpca,
        "tsne": Ttsne,
        "mds": Tmds,
        "umap": Tumap,
        "lda": Tlda,
        "dbscan": Tdbscan,
        "kmeans": Tkmeans,

        "null_nom1": Tnull,
        "null_nom": Tnull,
        "null_num1": Tnull,
        "null_num": Tnull,

        "test": Ttest
    }

    # process the basic transformation path
    pid = "r"
    cid = "r"
    if tpath not in [None, "", " ", [], {}]:
        for t in tpath:
            cid = pid + SEPERATION + str(t)

            ndata = Tbasic(ndata, t)

            if tpathtree is not None and ndata is not None:
                if tpathtree[cid].data is None:
                    tpathtree[cid].data = list(ndata.columns)
            pid = cid


    # process the core transformation
    if coret not in [None, "", " ", [], {}]:
        if NOTCALCUDMT:
            if coret["name"] in dmTl and sum([t["t"] in ["sum", "sub", "mul", "div"] for t in tpath]) > 0:
                ndata = None
            else:
                ndata = transformation_functions[coret["name"]](data=ndata, para=coret["para"])
        else:
            ndata = transformation_functions[coret["name"]](data=ndata, para=coret["para"])
        if coret["name"] == "null_num1":
            if sum([t["t"] in ["sum", "sub", "mul", "div"] for t in tpath]) == 0:
                ndata = None
        cid = pid + SEPERATION + str(coret)
        if tpathtree is not None and ndata is not None:
            if tpathtree[cid].data is None:
                tpathtree[cid].data = list(ndata.columns) if isinstance(ndata, pd.DataFrame) else ([ndata.name] if ndata.name is not None else [0])
    if tpathtree is not None:
        return ndata, tpathtree
    else:
        return ndata


def Tbasic(data, t):
    ndata = data

    # process input
    if t["i_type"] == "like":
        idata = ndata.select_dtypes(include=t["i"])
    elif t["i_type"] == "==":
        idata = ndata[t["i"]]
    elif t["i_type"] == "all":
        idata = ndata
    elif t["i_type"] == "num":
        idata = ndata.select_dtypes(include=["int", "float"])
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
    elif t["t"] == "sub":
        odata = pd.DataFrame(idata[idata.columns[0]]-idata[idata.columns[1]])
    elif t["t"] == "div":
        odata = pd.DataFrame(idata[idata.columns[0]]/idata[idata.columns[1]]).fillna(0)
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

    return ndata

def Tpca(data, para):
    if errorinputforcoreT(data):
        return None
    ndata = data.select_dtypes(include=["int", "float"])
    res = ppca(ndata, **para)   # numpy result

    return pd.DataFrame(res, columns=pd.Index(["PC1", "PC2"]))

def Ttsne(data, para):
    if errorinputforcoreT(data):
        return None
    ndata = data.select_dtypes(include=["int", "float"])
    res = ptsne(ndata, **para)  # numpy result

    return pd.DataFrame(res, columns=pd.Index(["tSNE-1", "tSNE-2"])).astype("float64")

def Tmds(data, para):
    if errorinputforcoreT(data):
        return None
    ndata = data.select_dtypes(include=["int", "float"])
    res = pmds(ndata, **para)  # numpy result

    return pd.DataFrame(res, columns=pd.Index(["MDS-1", "MDS-2"])).astype("float64")

def Tumap(data, para):
    if errorinputforcoreT(data):
        return None
    ndata = data.select_dtypes(include=["int", "float"])
    res = pumap(ndata, **para)  # numpy result

    return pd.DataFrame(res, columns=pd.Index(["UMAP-1", "UMAP-2"])).astype("float64")

def Tlda(data, para):
    if errorinputforcoreT(data):
        return None
    ndata = data.select_dtypes(include=["int", "float"])
    res = plda(ndata, **para)   # numpy result

    return pd.Series(res, name="Category by LDA")

def Tdbscan(data, para):
    if errorinputforcoreT(data):
        return None
    ndata = data.select_dtypes(include=["int", "float"])
    res = pdbscan(ndata, **para)   # numpy result

    return pd.Series(res, name="Category by DBSCAN")

def Tkmeans(data, para):
    if errorinputforcoreT(data):
        return None
    ndata = data.select_dtypes(include=["int", "float"])
    res = pkmeans(ndata, **para)   # numpy result

    return pd.Series(res, name="Category by KMeans").astype("int64")

def Tnull(data, para):
    ndata = data
    return ndata

def Tnullnum(data, para):
    ndata = data.select_dtypes(include=["int", "float"])
    return ndata

def Tnullcat(data, para):
    ndata = data.select_dtypes(include=["object"])
    return ndata

def Ttest(data, para):
    return data

def errorinputforcoreT(data):
    if not isinstance(data, pd.DataFrame) or len(data.columns) <= 2:
        return True
    return False

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