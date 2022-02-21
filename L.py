cal_load = {
    "sum": 10,
    "sub": 10,
    "mul": 50,
    "div": 40,
    "rank": 5,
    "del": 1,
    "select": 1,
    "astype": 2,
    "aggr": 100,
    "order": 10,
}


class Load:
    def __init__(self, l):
        self.l = l

    def __eq__(self, other):
        return self.l[0] == other.l[0] and self.l[1] == other.l[1] and self.l[2] == other.l[2]

    def __lt__(self, other):
        return self.l[0] < other.l[0] and self.l[1] < other.l[1] and self.l[2] < other.l[2]

    def __le__(self, other):
        return self.l[0] <= other.l[0] and self.l[1] <= other.l[1] and self.l[2] <= other.l[2]

    def __gt__(self, other):
        return self.l[0] > other.l[0] and self.l[1] > other.l[1] and self.l[2] > other.l[2]

    def __ge__(self, other):
        return self.l[0] >= other.l[0] and self.l[1] >= other.l[1] and self.l[2] >= other.l[2]

if __name__ == "__main__":
    print(Load((1,2,3)) > Load((1,2,2)))
    print(Load((1,2,3)) >= Load((1,2,2)))
    print(Load((1,2,3)) != Load((1,2,2)))
