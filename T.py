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
        "para": {
            "n_components": 2
        }
    },
    "lda": {
        "name": "lda",
        "input": {
            "dim": None,
            "type": "int"
        },
        "output": {
            "dim": 1,
            "type": "cat"
        },
        "para": {
            "n_components": 6
        }
    },
    "test": {
        "name": "test",
        "input": {
            "dim": None,
            "type": None
        },
        "output": {
            "dim": 1,
            "type": "cat"
        },
        "para": {}
    }
}

numtl = ["test", "pca"]
cattl = ["lda"]

class tpath(list):
    def __lt__(self, other):
        return len(self) < len(other)
