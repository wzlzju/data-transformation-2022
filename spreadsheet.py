import pandas as pd
import numpy as np

import os, sys
import json, csv

import utils




class spreadsheet:
    def __init__(self, datapath=None, **kwargs):
        self.datapath = datapath
        self.data = None
        if datapath:
            self.datapath = datapath
            if self.datapath.endswith(".json"):
                self.data = pd.read_json(self.datapath, **kwargs)
            elif self.datapath.endswith(".csv"):
                self.data = pd.read_csv(self.datapath, **kwargs)
            else:
                print("error: unacceptable data format")
                raise Exception("error data format")

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
        self.colinfo = {
            "col_names": self.columnnames,
            "col_type": {},
            "dist_mat": {},
            "col_names_simi": {}
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
                        self.colinfo["col_type"][col]["iskey"] = True
                        self.key = col
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

        # compute column names similarity
        self.colinfo["col_names_simi"]["vectors"] = utils.w2v(list(self.columnnames))
        self.colinfo["col_names_simi"]["cosine"] = pd.DataFrame(utils.distmat(self.colinfo["col_names_simi"]["vectors"],
                                                                              self.colinfo["col_names_simi"]["vectors"],
                                                                              metric="cosine"),
                                                                index=self.data.columns,
                                                                columns=self.data.columns)







if __name__ == "__main__":
    d1 = spreadsheet(datapath="./testdata/ie19.csv")
    print(d1.colinfo)
    d2 = spreadsheet(datapath="./testdata/NetflixOriginals.csv", encoding="unicode_escape")
    print(d2.colinfo)


