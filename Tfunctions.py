import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import pandas as pd

def ppca(data, n_components=2):
    pca = PCA(n_components=n_components)
    res = pca.fit_transform(data)
    return res

def plda(data, n_components=6):
    tdata = []
    for row in data.itertuples():
        tdata.append(" ".join([" ".join(["TOKEN" + str(i) + colname] * getattr(row, colname)) for i, colname in enumerate(data)]))
    cntVector = CountVectorizer()
    cvecdata = cntVector.fit_transform(tdata)
    lda = LatentDirichletAllocation(n_components=n_components)
    ldares = lda.fit_transform(cvecdata)
    res = np.argmax(ldares, axis=1)
    return res
