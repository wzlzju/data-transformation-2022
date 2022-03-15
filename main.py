import json
import os, csv
import base64

import pandas as pd
from flask import Flask, request
from copy import deepcopy
from gevent import pywsgi

from spreadsheet import spreadsheet
from search import searchobj

from config import *

import V
import T
import score


app = Flask(__name__)
app.debug = DEBUG

# global variants
sheet = None
sobj = None
stree = None
visdata = None

# add the CORS header
@app.after_request
def cors(environ):
    environ.headers['Access-Control-Allow-Origin']='*'
    environ.headers['Access-Control-Allow-Method']='*'
    environ.headers['Access-Control-Allow-Headers']='x-requested-with,content-type'
    if not DEBUG:
        print("\033[0;32mAfter request\033[0m")
    return environ

@app.route('/vis/csv', methods=['POST'])
def csv_in():
    datastr = request.get_data().decode("utf-8")
    data = json.loads(datastr)
    header = data.get("headers", "default")
    table = data.get("body", "default")
    print("headers", type(header), header)
    # print("body", type(table), table)
    global sheet, sobj, stree
    sheet = spreadsheet(dataframe=pd.DataFrame(table, columns=pd.Index(header)), encoding="unicode_escape", keep_default_na=False)
    ret = {
        "columns": {
            "headers": ["attribute", "type", "domain", "max", "min", "iskey", "values"],
            "body": []
        },
        "dim_clusters": [],
        "sem_clusters": []
    }

    for colname in sheet.columnnames:
        content = []
        content.append(colname) # attribute
        content.append(sheet.colinfo["col_type"][colname]["type"])  # type
        content.append(str(sheet.colinfo["col_type"][colname].get("domain", "")))    # domain
        content.append(str(sheet.colinfo["col_type"][colname].get("max", "")))   # max
        content.append(str(sheet.colinfo["col_type"][colname].get("min", "")))   # min
        content.append("T" if sheet.colinfo["col_type"][colname].get("iskey", False) else "")  # iskey
        content.append(", ".join([str(i) for i in sheet.data[colname].values]))
        ret["columns"]["body"].append(content)

    ret["dim_clusters"] = sheet.colinfo["dim_match"]["clusters"]
    ret["sem_clusters"] = sheet.colinfo["col_names_simi"]["clusters"]

    return json.dumps(ret)


@app.route('/vis/search', methods=['POST'])
def search_begin():
    datastr = request.get_data().decode("utf-8")
    data = json.loads(datastr)
    global sheet, sobj, stree, visdata
    # print(sheet.data)
    sobj = searchobj(dataobj=sheet)
    vl = data.get("vlist", ["scatter", "line", "bar"])
    tvl = []
    if "scatter" in vl:
        tvl += ["num_scatter", "cat_scatter"]
    if "line" in vl:
        tvl += ["ord_line", "ord_cat_line", "rel_line", "rel_cat_line"]
    if "bar" in vl:
        tvl += ["sum_bar", "count_bar"]
    ttl = data.get("tlist", T.tlist.keys())
    for tt in ["null_num", "null_num1", "null_nom", "null_nom1"]:
        if tt not in ttl:
            ttl.append(tt)
    sobj.configuration["vlist"] = tvl
    sobj.configuration["tlist"] = ttl
    sobj.configuration["slist"] = data.get("slist", score.slist)
    sobj.dataobj.colinfo["dim_match"]["clusters"] = data.get("dim_clusters", sobj.dataobj.colinfo["dim_match"]["clusters"])
    sobj.dataobj.colinfo["col_names_simi"]["clusters"] = data.get("sem_clusters", sobj.dataobj.colinfo["col_names_simi"]["clusters"])
    sobj.presearch()
    sobj.postsearchinitialization()
    stree = sobj.postsearch()
    visdata = sobj.assemblevisdata(round=1)
    sobj.assembleandevaluevis()
    ret = sobj.assembleTtree()
    # sobj.deconstruct()
    return json.dumps(ret)

@app.route('/vis/addT', methods=['POST'])
def addT():
    datastr = request.get_data().decode("utf-8")
    data = json.loads(datastr)
    global sheet, sobj, stree, visdata
    pid = data.get("pid", None)
    t = data.get("t", None)
    para = data.get("para", {})
    ret = sobj.singletransformation(pid, t, **para)
    return json.dumps({
        "result": ret,
        "highlight": ret["nodes"][-1]["id"]
    })

@app.route('/vis/addV', methods=['POST'])
def addV():
    datastr = request.get_data().decode("utf-8")
    data = json.loads(datastr)
    global sheet, sobj, stree, visdata
    vtype = data.get("vtype", None)
    channels = data.get("channels", None)
    ret = sobj.addvisualization(vtype, channels)
    return json.dumps({
        "result": ret,
        "highlight": ret["nodes"][-1]["id"]
    })


if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
	# production environment
    # server = pywsgi.WSGIServer((HOST, PORT), app)
    # server.serve_forever()
