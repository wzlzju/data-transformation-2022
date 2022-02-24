import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from umap import UMAP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import pandas as pd

def ppca(data, n_components=2):
    data = (data - data.min()) / (data.max() - data.min())
    pca = PCA(n_components=n_components)
    res = pca.fit_transform(data)
    return res

def ptsne(data, n_components=2):
    data = (data - data.min()) / (data.max() - data.min())
    tsne = TSNE(n_components=n_components)
    res = tsne.fit_transform(data)
    return res

def pmds(data, n_components=2):
    data = (data - data.min()) / (data.max() - data.min())
    mds = MDS(n_components=n_components, metric=True)
    res = mds.fit_transform(data)
    return res

def pumap(data, n_components=2):
    data = (data - data.min()) / (data.max() - data.min())
    umap = UMAP(n_components=n_components)
    res = umap.fit_transform(data)
    return res

def plida(data, n_components=2):
    X = data["X"]
    y = data["y"]
    lda = LinearDiscriminantAnalysis(n_components=n_components).fit(X, y)
    res = lda.transform(X)
    return res

def plda(data, n_components=4):
    td = pd.DataFrame()
    for col in data.columns:
        ts = data[col].rank(axis=0,method='first',numeric_only=True,na_option='keep',ascending=True,pct=False).astype("int64")
        td = pd.concat([td, ts], axis=1)
    data = td
    data.columns = pd.Index([clean_col_name(colname) for colname in list(data.columns)])
    tdata = []
    for row in data.itertuples():
        tdata.append(" ".join([" ".join(["TOKEN" + str(i) + colname] * getattr(row, colname)) for i, colname in enumerate(data)]))
    cntVector = CountVectorizer()
    cvecdata = cntVector.fit_transform(tdata)
    lda = LatentDirichletAllocation(n_components=n_components)
    ldares = lda.fit_transform(cvecdata)
    res = np.argmax(ldares, axis=1)
    return res

def clean_col_name(s):
    return (s[1:] if s.startswith(" ") else s).replace(':', '_').replace(',', "_").replace(';', '_').replace('.', '_').replace('?', '_').replace('/', '_')\
        .replace('*', '_').replace('!', '_').replace('(', '_').replace(')', '_').replace(' ', '_')

def pdbscan(data, eps=0.25, min_samples=5):
    data = (data - data.min()) / (data.max() - data.min())
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    res = dbscan.fit_predict(data)
    return res

def pkmeans(data, n_components=3):
    data = (data - data.min()) / (data.max() - data.min())
    kmeans = KMeans(n_clusters=n_components, random_state=9)
    res = kmeans.fit_predict(data)
    return res