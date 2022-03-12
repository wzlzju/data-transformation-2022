vlist = {
    "cat_scatter": {
        "name": "num_scatter",
        "input": {
            "xy": {
                "name": "xy",
                "dim": 2,
                "type": "num"
            },
            "color": {
                "name": "color",
                "dim": 1,
                "type": "cat"
            }
        }
    },
    "num_scatter": {
        "name": "num_scatter",
        "input": {
            "xy": {
                "name": "xy",
                "dim": 2,
                "type": "num"
            },
            "color": {
                "name": "color",
                "dim": 1,
                "type": "num"
            }
        }
    },
    "ord_line": {
        "name": "ord_line",
        "input": {
            "y": {
                "name": "y",
                "dim": None,
                "type": "num"
            }
        }
    },
    "ord_cat_line": {
        "name": "ord_cat_line",
        "input": {
            "y": {
                "name": "y",
                "dim": 1,
                "type": "cat"
            }
        }
    },
    "rel_line": {
        "name": "rel_line",
        "input": {
            "x": {
                "name": "x",
                "dim": 1,
                "type": "num"
            },
            "y": {
                "name": "y",
                "dim": None,
                "type": "num"
            }
        }
    },
    "rel_cat_line": {
        "name": "rel_cat_line",
        "input": {
            "x": {
                "name": "x",
                "dim": 1,
                "type": "num"
            },
            "y": {
                "name": "y",
                "dim": 1,
                "type": "cat"
            }
        }
    },
    "sum_bar": {
        "name": "sum_bar",
        "input": {
            "x": {
                "name": "x",
                "dim": 1,
                "type": "cat"
            },
            "y": {
                "name": "y",
                "dim": None,
                "type": "num"
            }
        }
    },
    "count_bar": {
        "name": "count_bar",
        "input": {
            "x": {
                "name": "x",
                "dim": 1,
                "type": "cat"
            },
            "y": {
                "name": "y",
                "dim": None,
                "type": None
            }
        }
    }
}