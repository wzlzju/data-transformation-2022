tlist = {
    "pca": {
        "name": "pca",
        "input": {
            "dim": None,
            "type": "num"
        },
        "output": {
            "dim": 2,
            "type": "num"
        },
        "para": {}
    },
    "tsne": {
        "name": "tsne",
        "input": {
            "dim": None,
            "type": "num"
        },
        "output": {
            "dim": 2,
            "type": "num"
        },
        "para": {}
    },
    "mds": {
        "name": "mds",
        "input": {
            "dim": None,
            "type": "num"
        },
        "output": {
            "dim": 2,
            "type": "num"
        },
        "para": {}
    },
    "umap": {
        "name": "umap",
        "input": {
            "dim": None,
            "type": "num"
        },
        "output": {
            "dim": 2,
            "type": "num"
        },
        "para": {}
    },
    "lida": {
        "name": "lida",
        "input": [{
            "dim": None,
            "type": "num"
        }, {
            "dim": 1,
            "type": "int"
        }],
        "output": {
            "dim": 2,
            "type": "num"
        },
        "para": {}
    },
    "lda": {
        "name": "lda",
        "input": {
            "dim": None,
            "type": "num"
        },
        "output": {
            "dim": 1,
            "type": "cat"
        },
        "para": {}
    },
    "dbscan": {
        "name": "dbscan",
        "input": {
            "dim": None,
            "type": "num"
        },
        "output": {
            "dim": 1,
            "type": "cat"
        },
        "para": {}
    },
    "kmeans": {
        "name": "kmeans",
        "input": {
            "dim": None,
            "type": "num"
        },
        "output": {
            "dim": 1,
            "type": "cat"
        },
        "para": {}
    },
    "null_num1": {
        "name": "null_num1",
        "input": {
            "dim": 1,
            "type": "num"
        },
        "output": {
            "dim": 1,
            "type": "num"
        },
        "para": {}
    },
    "null_num": {
        "name": "null_num1",
        "input": {
            "dim": 1,
            "type": "num"
        },
        "output": {
            "dim": None,
            "type": "num"
        },
        "para": {}
    },
    "null_nom1": {
        "name": "null_nom1",
        "input": {
            "dim": 1,
            "type": "nominal"
        },
        "output": {
            "dim": 1,
            "type": "cat"
        },
        "para": {}
    },
    "test": {
        "name": "test",
        "input": {
            "dim": None,
            "type": None
        },
        "output": {
            "dim": 1,
            "type": "num"
        },
        "para": {}
    }
}

numtl = ["pca", "tsne", "mds", "umap", "null_num", "null_num1"]
cattl = ["dbscan", "kmeans", "lda", "null_nom1"]
numtl = ["null_num"]
cattl = ["lda", "kmeans", "null_nom1"]

class tpath(list):
    def __lt__(self, other):
        return len(self) < len(other)


basicTl = ['rank', 'aggr', 'sum', 'sub', 'mul', 'div']
dmTl = ["pca", "tsne", "mds", "umap", "dbscan", "kmeans", "lda", "lida"]
alignTl = ["pca", "tsne", "mds", "umap", "dbscan", "kmeans", "lda", "null_num", "null_num1"]

threadsharing = [["pca", "tsne", "mds", "umap", "dbscan", "kmeans", "lda", "null_num"], ["lida"], ["null_nom1"], ["null_num1"], ["test"]]
def getRepT(tname):
    for ss in threadsharing:
        if tname in ss:
            for tht in ss:
                if tht in numtl+cattl:
                    return tht
    return None
