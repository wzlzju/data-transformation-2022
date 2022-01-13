import json, csv
import os, sys
from copy import deepcopy


def ie2csv(path):
    with open("./testdata/countries.json", "r") as f:
        countries = json.load(f)
    with open(path, "r") as f:
        d = json.load(f)
    with open(path[:-5]+".csv", "w", newline="", encoding="utf-8") as f:
        spamwriter = csv.writer(f)#, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["country"]+["exp"+str(i) for i in range(10)]+["imp"+str(i) for i in range(10)])
        for ci in d.keys():
            c = d[ci]
            spamwriter.writerow([countries[ci]["country_name_abbreviation"]]+c["exp"]+c["imp"])




if __name__ == "__main__":
    ie2csv("./testdata/ie95.json")