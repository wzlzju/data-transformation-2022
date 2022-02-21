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
    header = data.get("header", "default")
    table = data.get("table", "default")
    print("header", type(header), header)
    print("table", type(table), table)
    global sheet, sobj, stree
    sheet = spreadsheet(dataframe=pd.DataFrame(table, columns=pd.Index(header)), encoding="unicode_escape", keep_default_na=False)
    ret = {
        "columns": {
            "header": ["attribute", "type", "domain", "max", "min", "iskey", "values"],
            "content": []
        },
        "dim_clusters": [],
        "sem_clusters": []
    }

    for colname in sheet.columnnames:
        content = []
        content.append(colname) # attribute
        content.append(sheet.colinfo["col_type"][colname]["type"])  # type
        content.append(str(sheet.colinfo["col_type"][colname]["domain"]))    # domain
        content.append(sheet.colinfo["col_type"][colname].get("max", ""))   # max
        content.append(sheet.colinfo["col_type"][colname].get("min", ""))   # min
        content.append("T" if sheet.colinfo["col_type"][colname]["iskey"] else "")  # iskey
        content.append(", ".join([str(i) for i in sheet.data[colname].values]))
        ret["columns"]["content"].append(content)

    ret["dim_clusters"] = sheet.colinfo["dim_match"]["clusters"]
    ret["sem_clusters"] = sheet.colinfo["col_names_simi"]["clusters"]

    return json.dumps(ret)


@app.route('/vis/search', methods=['GET'])
def search_begin():
    global sheet, sobj, stree, visdata
    sobj = searchobj(dataobj=sheet)
    sobj.presearch()
    sobj.postsearchinitialization()
    stree = sobj.postsearch()
    visdata = sobj.assemblevisdata(round=1)


if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
	# production environment
    # server = pywsgi.WSGIServer((HOST, PORT), app)
    # server.serve_forever()
