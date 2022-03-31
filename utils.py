import re

import numpy as np
import pandas as pd
from datetime import date, datetime
from scipy.spatial.distance import cdist
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from gensim.models import KeyedVectors
from gensim.parsing import preprocessing
from copy import deepcopy

w2v_model = KeyedVectors.load_word2vec_format("./external/word2vec-slim/GoogleNews-vectors-negative300-SLIM.bin", binary=True)

def tocontinue(ii, idx, label):
    if idx is not None:
        if isinstance(idx, dict):
            if idx.get(label, None) is not None:
                if isinstance(idx[label], list):
                    if ii not in idx[label] and str(ii) not in idx[label]:
                        return True
                else:
                    if ii != idx[label] and str(ii) != idx[label]:
                        return True
        elif isinstance(idx, list):
            if ii not in idx and str(ii) not in idx:
                return True
        else:
            if ii != idx and str(ii) != idx:
                return True
    return False

def dellistelements(a, b):
    # return list(set(a) - set(b))
    t = deepcopy(a)
    for i in b:
        t.remove(i)
    return t

def listintersection(a, b):
    # return list(set(a) & set(b))
    return [i for i in a if i in b]

def isdate(entry):
    """
    from https://github.com/nl4dv/nl4dv
    :param entry: input data entry (str-like)
    :return: True/False, datetime(obj)/None
    """
    date_regexes = [
        # Format:
            # MM*DD*YY(YY) where * ∈ {, . - /}
        # Examples:
            # 12.24.2019
            # 12:24:2019
            # 12-24-2019
            # 12/24/2019
            # 1/24/2019
            # 07/24/2019
            # 1/24/20
        [['%m/%d/%Y', '%m/%d/%y'], "([1][0-2]|[0]?[1-9])[-,.\/]+([1|2][0-9]|[3][0|1]|[0]?[1-9])[-,.\/]+([1-9][0-9]{3}|[0-9]{2})"],
        # Format:
            # YY(YY)*MM*DD where * ∈ {, . - /}
        # Examples:
            # 2019.12.24
            # 2019.12.24
            # 2019-12-24
            # 2019/12/24
            # 2019/1/24
            # 2019/07/24
            # 20/1/24
        [['%Y/%m/%d', '%y/%m/%d'], "([1-9][0-9]{3}|[0-9]{2})[-,.\/]+([1][0-2]|[0]?[1-9])[-,.\/]+([1|2][0-9]|[3][0|1]|[0]?[1-9])"],
        # Format:
            # DD*MM*YY(YY) where * ∈ {, . - /}
        # Examples:
            # 24.12.2019
            # 24:12:2019
            # 24-12-2019
            # 24/12/2019
            # 24/1/2019
            # 24/07/2019
            # 24/1/20
        [['%d/%m/%Y', '%d/%m/%y'], "([1|2][0-9]|[3][0|1]|[0]?[1-9])[-,.\/]+([1][0-2]|[0]?[1-9])[-,.\/]+([1-9][0-9]{3}|[0-9]{2})"],
        # Formats:
            # DD*MMM(M)*YY(YY) where * ∈ {, . - / <space>}
        # Examples:
            # 8-January-2019
            # 31 Dec 19
        [['%d/%b/%Y', '%d/%B/%Y', '%d/%b/%y', '%d/%B/%y'], "([1|2][0-9]|[3][0|1]|[0]?[1-9])[-,.\/\s]+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-,.\/\s]+([1-9][0-9]{3}|[0-9]{2})"],
        # Format:
            # DD*MMM(M) where * ∈ {, . - / <space>}
        # Examples:
            # 31-January
            # 1 Jul
        [['%d/%b', '%d/%B'], "([1|2][0-9]|[3][0|1]|[0]?[1-9])[-,.\/\s]+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"],
        # Formats:
            # MMM(M)*DD*YYY(Y) where * ∈ {, . - / <space>}
        # Examples:
            # January-8-2019
            # Dec 31 19
        [['%b/%d/%Y', '%B/%d/%Y', '%b/%d/%y', '%B/%d/%y'], "(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-,.\/\s]+([1|2][0-9]|[3][0|1]|[0]?[1-9])[-,.\/\s]+([1-9][0-9]{3}|[0-9]{2})"],
        # Format:
            # MMM(M)*DD where * ∈ {, . - / <space>}
        # Examples:
            # January-31
            # Jul 1
        [['%b/%d', '%B/%d'], "(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-,.\/\s]+([1|2][0-9]|[3][0|1]|[0]?[1-9])"],
        # Format:
            # YYYY
        # Examples:
            # 18XX, 19XX, 20XX
        [["%Y"], "(1[8-9][0-9][0-9]|20[0-2][0-9])"]
    ]

    try:
        for idx, regex_list in enumerate(date_regexes):
            regex = re.compile(regex_list[1])
            match = regex.match(str(entry))
            if match is not None:
                for f in regex_list[0]:
                    try:
                        dateobj = datetime.strptime("/".join(list(match.groups())), f)
                        return True, dateobj
                    except Exception as e:
                        dateobj = None
                return False,  None
        return False, None
    except Exception as e:
        return False, None

def dist(x, y, metric="euclidean", *args, **kwargs):
    if metric == "wasserstein":
        return wasserstein_distance(x, y)
    else:
        return cdist([x], [y], metric)[0][0]

def distmat(x, y, metric="euclidean", *args, **kwargs):
    if isinstance(x, list) and isinstance(y, list):
        if metric == "wasserstein":
            return np.array([[wasserstein_distance(x[i], y[j]) for j in range(len(y))] for i in range(len(x))])
        else:
            return cdist(x, y, metric)
    elif isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        if metric == "wasserstein":
            if kwargs["type"][:3] == "col":
                tmpnp = np.array([[wasserstein_distance(x[i], y[j]) for j in y] for i in x])
            else:
                xx = pd.DataFrame(x.values.T, index=x.columns, columns=x.index)
                yy = pd.DataFrame(y.values.T, index=y.columns, columns=y.index)
                tmpnp = np.array([[wasserstein_distance(xx[i], yy[j]) for j in yy] for i in xx])
        else:
            if kwargs["type"][:3] == "col":
                xx = pd.DataFrame(x.values.T, index=x.columns, columns=x.index)
                yy = pd.DataFrame(y.values.T, index=y.columns, columns=y.index)
                tmpnp = cdist(xx, yy, metric)
            else:
                tmpnp = cdist(x, y, metric)
        if kwargs["type"][:3] == "col":
            retpd = pd.DataFrame(tmpnp, index=x.columns, columns=y.columns)
        else:
            retpd = pd.DataFrame(tmpnp, index=x.index, columns=y.index)

        return retpd

def sproc(s):
    filters = [preprocessing.lower_to_unicode,
               preprocessing.strip_non_alphanum,
               preprocessing.strip_punctuation,
               preprocessing.split_alphanum,
               preprocessing.strip_numeric,
               preprocessing.strip_tags,
               preprocessing.strip_multiple_whitespaces,
               preprocessing.remove_stopwords]
    return preprocessing.preprocess_string(s, filters)

def w2v(words, stype="single"):
    token = 'TOKEN'

    def _w2v(ss):
        return w2v_model[ss] if ss in w2v_model else w2v_model[token]

    if stype == "single":
        def _swp(ss):
            return sproc(ss)[0] if len(sproc(ss)) > 0 else token

        if isinstance(words, str):
            return _w2v(_swp(words))
        elif isinstance(words, list) and isinstance(words[0], str):
            return [_w2v(_swp(w)) for w in words]
        elif isinstance(words, list):
            print("error: unexpected type:", type(words), type(words[0]), "in utils <func w2v><branch single>")
            raise Exception("error unexpected type")
        else:
            print("error: unexpected type:", type(words), "in utils <func w2v><branch single>")
            raise Exception("error unexpected type")
    elif stype == "multiple":
        def _mwp(ss):
            return sproc(ss) if len(sproc(ss)) > 0 else [token]

        if isinstance(words, str):
            single_words = _mwp(words)
            return sum([_w2v(w) for w in single_words]) / len(single_words)
        elif isinstance(words, list) and isinstance(words[0], str):
            return [sum([_w2v(w) for w in _mwp(sen)]) / len(_mwp(sen)) for sen in words]
        elif isinstance(words, list):
            print("error: unexpected type:", type(words), type(words[0]), "in utils <func w2v><branch multiple>")
            raise Exception("error unexpected type")
        else:
            print("error: unexpected type:", type(words), "in utils <func w2v><branch multiple>")
            raise Exception("error unexpected type")
    else:
        print("error: unexpected type:", stype, "in utils <func w2v>")
        raise Exception("error unexpected type")

def decorate(r):
    ret = r
    beautifylist = [0, 2, 3, 6, 12] if len(ret["vis_list"]) >= 13 else [3, 2, 0, 4, 8, 7]
    for i in range(len(ret["vis_list"])):
        if i not in beautifylist:
            beautifylist.append(i)
    ret["vis_list"] = [ret["vis_list"][i] for i in beautifylist if i < len(ret["vis_list"])]
    return ret

def mean(l):
    return sum(l) / len(l) if len(l) > 0 else 0

def Tstr2obj(s):
    return eval(s.replace("Index", "pd.Index").replace("array", "np.array"))

def suml(l):
    if len(l) == 0:
        return l
    ret = l[0]
    for i in range(1, len(l)):
        ret += l[i]
    return ret

if __name__ == "__main__":
    d1 = pd.read_csv("./testdata/NetflixOriginals.csv", encoding="unicode_escape")
    print(isdate(d1['Premiere'][0]))
    d2 = pd.read_csv("./testdata/ie19.csv")
    d = d2.select_dtypes(include=['int', 'float'])
    print([dist(d['exp'+str(i)], d['imp'+str(i)], "wasserstein") for i in range(10)], type(dist(d['exp'+str(0)], d['imp'+str(0)], "wasserstein")))
    print([dist(d['exp'+str(0)], d['exp'+str(i)], "wasserstein") for i in range(10)])
    print([dist(d['exp'+str(i)], d['imp'+str(i)], "jensenshannon") for i in range(10)], type(dist(d['exp'+str(0)], d['imp'+str(0)], "jensenshannon")))
    print([dist(d['exp'+str(0)], d['exp'+str(i)], "jensenshannon") for i in range(10)])
    print(distmat([d['exp0'], d['exp1'], d['exp2']], [d['imp0'], d['imp1']], "wasserstein"), type(distmat([d['exp0'], d['exp1'], d['exp2']], [d['imp0'], d['imp1']], "wasserstein")))
    print(distmat([d['exp0'], d['exp1'], d['exp2']], [d['imp0'], d['imp1']], "jensenshannon"), type(distmat([d['exp0'], d['exp1'], d['exp2']], [d['imp0'], d['imp1']], "jensenshannon")))
    print(distmat(d[d.columns[:3]], d[d.columns[:2]], "wasserstein", type="col"))
    print(distmat(d[d.columns[:3]], d[d.columns[:2]], "wasserstein", type="row"))
    print(distmat(d[d.columns[:3]], d[d.columns[:2]], "jensenshannon", type="col"))
    try:
        print(distmat(d[d.columns[:3]], d[d.columns[:2]], "jensenshannon", type="row"))
    except ValueError as e:
        print("ValueError:", e)
    print(distmat(d1.select_dtypes(include=['int', 'float']), d1.select_dtypes(include=['int', 'float']), "wasserstein", type="col"))
    print(distmat(d1.select_dtypes(include=['int', 'float']), d1.select_dtypes(include=['int', 'float']), "jensenshannon", type="col"))
    column_names_vectors = w2v(list(d.columns))
    print(distmat(column_names_vectors, column_names_vectors, metric="cosine"))
    column_names_vectors = w2v(list(d2.columns))
    dp = pd.DataFrame(distmat(column_names_vectors, column_names_vectors, metric="cosine"), index=d2.columns, columns=d2.columns)
    print(dp)
    column_names_vectors = w2v(list(d1.columns))
    dp = pd.DataFrame(distmat(column_names_vectors, column_names_vectors, metric="cosine"), index=d1.columns, columns=d1.columns)
    print(dp)
