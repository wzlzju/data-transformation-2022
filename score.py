import numpy as np
import matplotlib.pyplot as plt


def getHist(data, label=None):
    # data: [idx, 2]
    min_x, max_x, min_y, max_y = data[:, 0].min(), data[:, 0].max() + 1, data[:, 1].min(), data[:, 1].max() + 1

    data[:, 0] = 100 * (data[:, 0] - min_x) / (max_x - min_x)
    data[:, 1] = 100 * (data[:, 1] - min_y) / (max_y - min_y)

    if label is None:
        hist = np.zeros((10, 10))

        for datum in data:
            hist[int(datum[0] // 10), int(datum[1] // 10)] += 1

        # normalize to [0, 1]
        hist = hist / data.shape[0]
        return hist
    else:
        classes = np.unique(label)
        hists = []
        for cls in classes:
            if cls < 0:
                continue
            hists.append(np.zeros((10, 10)))
            for datum in data[label == cls]:
                hists[-1][min(int(datum[0] // 10), 9), min(int(datum[1] // 10), 9)] += 1

            # normalize to [0, 1]
            # hists[-1] = hists[-1] / data[label == cls].shape[0]
        return hists


def CDM(data, label):
    hist_by_class = getHist(data, label=label)

    result = 0
    for i in range(len(hist_by_class)):
        for j in range(i + 1, len(hist_by_class)):
            result += np.sum(np.abs(hist_by_class[i] - hist_by_class[j]))

    result = result / data.shape[0] / (len(hist_by_class) - 1)
    return result * 100

class dotGraph:
    def __init__(self, dots, cache=True):
        self.dot_num = dots.shape[0]
        self.dots = dots
        if cache:
            self.cache = {}
        else:
            self.cache = None

    def dist(self, a, b):
        if self.cache is not None:
            if (min(a, b), max(a, b)) in self.cache.keys():
                return self.cache[(min(a, b), max(a, b))]
            else:
                self.cache[(min(a, b), max(a, b))] = np.sqrt(np.sum((self.dots[a] - self.dots[b]) ** 2))
                return self.cache[(min(a, b), max(a, b))]
        else:
            return np.sqrt(np.sum((a - b) ** 2))

    def cosineDist(self, vec, a, b, norma, normb):
        v1 = self.dots[a] - self.dots[vec]
        v2 = self.dots[b] - self.dots[vec]
        res = np.sum(v1 * v2) / (norma * normb)
        return res

    def minSpanTree(self):
        self.dot_in_tree = [0 for i in range(self.dot_num)]
        self.dot_in_tree[0] = 1
        self.dot_degree = [0 for i in range(self.dot_num)]
        self.treeEdge = {}  # edge with distance
        self.sorted_trrEdge = None # edge with distance sorted by distance
        self.edgeDict = {}  # adjacent map

        while sum(self.dot_in_tree) < self.dot_num:
            min_edge = None
            new_dot_in_tree = None
            min_edge_len = None
            for i in range(self.dot_num):
                if self.dot_in_tree[i] == 0:
                    continue
                for j in range(self.dot_num):
                    if self.dot_in_tree[j] == 1:
                        continue
                    else:
                        if min_edge_len is None or self.dist(i, j) < min_edge_len:
                            min_edge = (min(i, j), max(i, j))
                            new_dot_in_tree = j
                            min_edge_len = self.dist(i, j)
            self.dot_in_tree[new_dot_in_tree] = 1
            self.dot_degree[min_edge[0]] += 1
            self.dot_degree[min_edge[1]] += 1
            self.treeEdge[min_edge] = min_edge_len
            if min_edge[0] in self.edgeDict:
                self.edgeDict[min_edge[0]].append(min_edge[1])
            else:
                self.edgeDict[min_edge[0]] = [min_edge[1]]
            if min_edge[1] in self.edgeDict:
                self.edgeDict[min_edge[1]].append(min_edge[0])
            else:
                self.edgeDict[min_edge[1]] = [min_edge[0]]

        self.sorted_treeEdge = sorted(self.treeEdge.items(), key=lambda x: x[1], reverse=False)
        edge_num = len(self.sorted_treeEdge)
        # percentile of the MST edge lengths
        self.q75 = self.sorted_treeEdge[int(edge_num * 0.75)][1]
        self.q25 = self.sorted_treeEdge[int(edge_num * 0.25)][1]
        self.q90 = self.sorted_treeEdge[int(edge_num * 0.9)][1]
        self.q50 = self.sorted_treeEdge[int(edge_num * 0.5)][1]
        self.q10 = self.sorted_treeEdge[int(edge_num * 0.1)][1]
        return 0

    def bfs(self, start, mark_dot = None, searched_dot_count=False):
        que = []
        dot_searched = [0 for i in range(self.dot_num)]
        dot_searched[start] = 1
        if mark_dot is not None:
            for dot in mark_dot:
                dot_searched[dot] = 1
        que.append((start, 0))
        latest_searched = None
        while len(que) > 0:
            latest_searched, cur_dist = que.pop(0)
            if latest_searched not in self.edgeDict.keys():
                continue
            for neigh in self.edgeDict[latest_searched]:
                if dot_searched[neigh] == 1:
                    continue
                else:
                    que.append(
                        (neigh, cur_dist + self.treeEdge[(min(latest_searched, neigh), max(latest_searched, neigh))]))
                    dot_searched[neigh] = 1
        if searched_dot_count:
            return latest_searched, cur_dist, sum(dot_searched) - (0 if mark_dot is None else len(mark_dot))
        else:
            return latest_searched, cur_dist

    def diameter(self):
        furthest, _ = self.bfs(0)
        furfurthest, diameter = self.bfs(furthest)
        return diameter, furthest, furfurthest

    def stringy_value(self):
        # include outliers
        diameter, _, _ = self.diameter()
        length = 0
        for e in self.treeEdge.items():
            length += e[1]

        return 100 * diameter / length

    def straight_value(self):
        # include outliers
        diameter, a, b = self.diameter()
        dist = self.dist(a, b)
        return 100 * dist / diameter

    def outlying_value(self):
        w = self.q75 + 1.5 * (self.q75 - self.q25)
        cut_edges_length = 0
        tree_edges_length = 0
        for e in self.treeEdge.items():
            tree_edges_length += e[1]
            if e[1] > w:
                if self.dot_degree[e[0][0]] == 1 or self.dot_degree[e[0][1]] == 1:
                    cut_edges_length += e[1]

        return 100 * (tree_edges_length - cut_edges_length) / tree_edges_length

    def skew_value(self):
        return 100 * (self.q90 - self.q50) / (self.q90 - self.q10)

    def striated_value(self):
        angles = {}
        for i in range(self.dot_num):
            if self.dot_degree[i] == 2:
                angles[i] = []

        for e in self.treeEdge.items():
            if e[0][0] in angles.keys():
                angles[e[0][0]].append((e[0][1], e[1]))
            if e[0][1] in angles.keys():
                angles[e[0][1]].append((e[0][0], e[1]))

        res = 0
        for ag in angles.items():
            res += np.abs(self.cosineDist(ag[0], ag[1][0][0], ag[1][1][0], ag[1][0][1], ag[1][1][1]))
        res = res / len(angles)
        return 100 * res

    def spearman_value(self):
        import scipy.stats as stats
        r, _ = stats.spearmanr(self.dots)
        return 100 * abs(r)

    def clumpy_value(self):
        w = 0 * self.q50
        max_after_cut = -1
        for i in range(1, len(self.sorted_treeEdge) + 1):
            e = self.sorted_treeEdge[-i]
            if e[1] > w:
                _, _, subset_dots = self.bfs(e[0][0], mark_dot=[e[0][1]], searched_dot_count=True)
                alpha = 0.05
                if alpha * self.dot_num < subset_dots < (1-alpha) * self.dot_num:
                    max_after_cut = -i - 1
                else:
                    break
            else:
                break
        return 100 * (1 - self.sorted_treeEdge[max_after_cut][1] / self.sorted_treeEdge[-1][1])

if __name__ == "__main__":
    data = np.random.rand(100, 2)
    label = np.random.randint(2, size=100)
    for i in range(100):
        data[i] = data[i] + 1.1 * label[i]

    # data = np.random.rand(200, 2)
    # data[49] = data[49] + 3
    g = dotGraph(data)
    g.minSpanTree()
    print(g.outlying_value())
    print(g.skew_value())
    print(g.striated_value())
    print(g.stringy_value())
    print(g.straight_value())
    print(g.spearman_value())
    print(g.clumpy_value())
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
