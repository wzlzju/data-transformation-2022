import pandas as pd
import numpy as np

import os, sys
import json, csv

from config import *
import utils
from copy import deepcopy

from sklearn.cluster import DBSCAN

from scipy.stats import wasserstein_distance


class spreadsheet:
    def __init__(self, datapath=None, dataframe=None, **kwargs):
        self.datapath = datapath
        self.data = None
        if datapath is not None:
            self.datapath = datapath
            if self.datapath.endswith(".json"):
                self.data = pd.read_json(self.datapath, **kwargs)
            elif self.datapath.endswith(".csv"):
                self.data = pd.read_csv(self.datapath, **kwargs)
            else:
                print("error: unacceptable data format")
                raise Exception("error data format")
        elif dataframe is not None:
            self.data = dataframe

        if len(self.data) > MAXSOURCEDATAROWS:
            self.data = self.data.head(MAXSOURCEDATAROWS)
        self.columnnames = None
        self.dtypes = None
        self.colinfo = None
        self.shape = self.data.shape
        self.rowsnum = self.shape[0]
        self.colsnum = self.shape[1]
        self.key = None
        if self.rowsnum == 0 or self.colsnum == 0:
            return
        self.parsedata()


    def parsedata(self):
        self.columnnames = self.data.columns
        self.dtypes = self.data.dtypes
        self.num_data = self.data.select_dtypes(include=['int', 'float'])
        self.colinfo = {
            "col_names": self.columnnames,
            "num_col_names": self.num_data.columns,
            "col_type": {},
            "dist_mat": {},
            "col_names_simi": {},
            "dim_match": {},
            "new_col": [],
        }

        # parse columns type
        for idx, col in enumerate(self.columnnames):
            cc = self.data[col]
            if self.dtypes[idx] == "object":
                # judge whether date
                if utils.isdate(cc[0])[0]:
                    self.colinfo["col_type"][col] = {
                        "type": "date",
                        "max": None,
                        "min": None,
                        "date_value": []
                    }
                    for _, entry in enumerate(cc):
                        _, cd = utils.isdate(entry)
                        if not self.colinfo["col_type"][col]["max"]:
                            self.colinfo["col_type"][col]["max"] = cd
                        elif cd > self.colinfo["col_type"][col]["max"]:
                            self.colinfo["col_type"][col]["max"] = cd
                        if not self.colinfo["col_type"][col]["min"]:
                            self.colinfo["col_type"][col]["min"] = cd
                        elif cd < self.colinfo["col_type"][col]["min"]:
                            self.colinfo["col_type"][col]["min"] = cd
                        self.colinfo["col_type"][col]["date_value"].append(cd)
                else:
                    self.colinfo["col_type"][col] = {
                        "type": "str",
                        "domain": set(cc),
                        "iskey": False
                    }
                    if len(self.colinfo["col_type"][col]["domain"]) == self.rowsnum:
                        if self.key and self.colinfo["col_type"][self.key]["type"] in ["str"]:
                            pass
                        else:
                            if self.key:
                                self.colinfo["col_type"][self.key]["iskey"] = False
                            self.colinfo["col_type"][col]["iskey"] = True
                            self.key = col
                    elif len(self.colinfo["col_type"][col]["domain"]) <= self.rowsnum * NOMINALSTD:
                        self.colinfo["col_type"][col]["type"] = "nominal"
            elif self.dtypes[idx] == "int64":
                self.colinfo["col_type"][col] = {
                    "type": "int",
                    "domain": set(cc),
                    "max": None,
                    "min": None,
                    "iskey": False
                }
                for _, entry in enumerate(cc):
                    cd = int(entry)
                    if not self.colinfo["col_type"][col]["max"]:
                        self.colinfo["col_type"][col]["max"] = cd
                    elif cd > self.colinfo["col_type"][col]["max"]:
                        self.colinfo["col_type"][col]["max"] = cd
                    if not self.colinfo["col_type"][col]["min"]:
                        self.colinfo["col_type"][col]["min"] = cd
                    elif cd < self.colinfo["col_type"][col]["min"]:
                        self.colinfo["col_type"][col]["min"] = cd
                if len(self.colinfo["col_type"][col]["domain"]) == self.rowsnum:
                    if self.key and self.colinfo["col_type"][self.key]["type"] in ["str", "int"]:
                        pass
                    else:
                        if self.key:
                            self.colinfo["col_type"][self.key]["iskey"] = False
                        self.colinfo["col_type"][col]["iskey"] = True
                        self.key = col
            elif self.dtypes[idx] == "float64":
                self.colinfo["col_type"][col] = {
                    "type": "real",
                    "domain": set(cc),
                    "max": None,
                    "min": None,
                    "iskey": False
                }
                for _, entry in enumerate(cc):
                    cd = float(entry)
                    if not self.colinfo["col_type"][col]["max"]:
                        self.colinfo["col_type"][col]["max"] = cd
                    elif cd > self.colinfo["col_type"][col]["max"]:
                        self.colinfo["col_type"][col]["max"] = cd
                    if not self.colinfo["col_type"][col]["min"]:
                        self.colinfo["col_type"][col]["min"] = cd
                    elif cd < self.colinfo["col_type"][col]["min"]:
                        self.colinfo["col_type"][col]["min"] = cd
                if len(self.colinfo["col_type"][col]["domain"]) == self.rowsnum:
                    if self.key and self.colinfo["col_type"][self.key]["type"] in ["str", "int", "real"]:
                        pass
                    else:
                        if self.key:
                            self.colinfo["col_type"][self.key]["iskey"] = False
                        self.colinfo["col_type"][col]["iskey"] = True
                        self.key = col
            else:
                print("error: unsolved dtype:", self.dtypes[idx])
                raise Exception("error unsolved dtypes")

        # compute columns distribution distance matrix
        self.colinfo["dist_mat"]["wasserstein"] = utils.distmat(self.data.select_dtypes(include=['int', 'float']),
                                                                self.data.select_dtypes(include=['int', 'float']),
                                                                metric="wasserstein",
                                                                type="col")
        self.colinfo["dist_mat"]["jensenshannon"] = utils.distmat(self.data.select_dtypes(include=['int', 'float']),
                                                                self.data.select_dtypes(include=['int', 'float']),
                                                                metric="jensenshannon",
                                                                type="col")

        # pre-calculate dimension matching column clustering
        clustering_res = DBSCAN(eps=1, # need to be tuned
                                min_samples=5,
                                metric=wasserstein_distance,
                                metric_params=None,
                                algorithm='auto',
                                leaf_size=30,
                                p=None,
                                n_jobs=1).fit_predict(pd.DataFrame(self.num_data.values.T, index=self.num_data.columns, columns=self.num_data.index))
        self.colinfo["dim_match"]["clusters"] = []
        for i, cid in enumerate(clustering_res):
            if cid < 0:
                continue
            while cid >= len(self.colinfo["dim_match"]["clusters"]):
                self.colinfo["dim_match"]["clusters"].append([])
            self.colinfo["dim_match"]["clusters"][cid].append(self.num_data.columns[i])
        if list(self.colinfo["num_col_names"]) not in self.colinfo["dim_match"]["clusters"]:
            self.colinfo["dim_match"]["clusters"].append(list(self.colinfo["num_col_names"]))

        # compute column names similarity
        self.colinfo["col_names_simi"]["vectors"] = utils.w2v(list(self.columnnames))
        self.colinfo["col_names_simi"]["cosine"] = pd.DataFrame(utils.distmat(self.colinfo["col_names_simi"]["vectors"],
                                                                              self.colinfo["col_names_simi"]["vectors"],
                                                                              metric="cosine"),
                                                                index=self.data.columns,
                                                                columns=self.data.columns)

        # pre-calculate semantic column clustering
        clustering_res = DBSCAN(eps=0.5,
               min_samples=5,
               metric='euclidean',
               metric_params=None,
               algorithm='auto',
               leaf_size=30,
               p=None,
               n_jobs=1).fit_predict(self.colinfo["col_names_simi"]["vectors"])
        self.colinfo["col_names_simi"]["clusters"] = []
        for i, cid in enumerate(clustering_res):
            if cid < 0:
                continue
            while cid >= len(self.colinfo["col_names_simi"]["clusters"]):
                self.colinfo["col_names_simi"]["clusters"].append([])
            self.colinfo["col_names_simi"]["clusters"][cid].append(self.colinfo['col_names'][i])





if __name__ == "__main__":
    d1 = spreadsheet(datapath="./testdata/ie19.csv")
    #print(d1.colinfo)
    #d2 = spreadsheet(datapath="./testdata/NetflixOriginals.csv", encoding="unicode_escape")
    #print(d2.colinfo)


