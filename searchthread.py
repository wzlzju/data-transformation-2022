import json, csv
import os, sys, time
from copy import deepcopy
import pandas as pd

from multiprocessing import Process, Pool, Queue
from _queue import Empty
from queue import Full

from spreadsheet import spreadsheet
from V import *
from T import *
from Tfunctions import *
from config import *

def tpaththreadfunction(tname, colinfo, q):
    t = tlist[tname]
    tinput = t["input"]
    tinputdim = tinput["dim"]
    tinputtype = tinput["type"]
    colnames = colinfo["col_names"]
    coltype = colinfo["col_type"]
    distmat = colinfo["dist_mat"]
    colnamessimi = colinfo["col_names_simi"]
    colnamesvectors = colnamessimi["vectors"]
    simimat = colnamessimi["cosine"]
    pool = []

    def updatequeue():
        if not q.full():
            try:
                q.put(pool, block=True, timeout=0.001)
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
        tpp.append({
            "t": "astype",
            "i_type": "like",
            "i": ["int", "float"],
            "o_type": "new_table",
            "args": ("float64", ),
            "kwargs": {},
            "index": "default"
        })
        while True:
            updatequeue()
    elif tname == "pca":
        pool.append((0, tpath()))
        tpp = pool[0][1]
        while True:
            updatequeue()
    elif tname == "lda":
        pool.append((0, tpath()))
        tpp = pool[0][1]
        while True:
            updatequeue()

    if tinputdim is None and tinputtype == "int":
        # e.g., lda
        pass
    elif tinputdim is None and tinputtype == "num":
        # e.g., pca
        pass
