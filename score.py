import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgraph
import copy


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

class sciGraph:
    def __init__(self, dots):
        self.dot_num = dots.shape[0]
        self.dots = dots
        self.cache = {}
        self.adjmax = csr_matrix([[self.dist(i, j) for j in range(self.dot_num)]for i in range(self.dot_num)])

    def cosineDist(self, vec, a, b, norma, normb):
        v1 = self.dots[a] - self.dots[vec]
        v2 = self.dots[b] - self.dots[vec]
        res = np.sum(v1 * v2) / (norma * normb)
        return res

    def dist(self, a, b):
        if (min(a, b), max(a, b)) in self.cache.keys():
            return self.cache[(min(a, b), max(a, b))]
        else:
            self.cache[(min(a, b), max(a, b))] = np.sqrt(np.sum((self.dots[a] - self.dots[b]) ** 2))
            return self.cache[(min(a, b), max(a, b))]

    def minSpanTree(self):
        self.treeAdjmax = csgraph.minimum_spanning_tree(self.adjmax)
        self.vertex = np.array(self.treeAdjmax.nonzero())

        # percentile of the MST edge lengths
        tmp = self.treeAdjmax.toarray()[self.treeAdjmax.toarray() != 0]
        self.q75 = np.percentile(tmp, 75)
        self.q25 = np.percentile(tmp, 25)
        self.q90 = np.percentile(tmp, 90)
        self.q50 = np.percentile(tmp, 50)
        self.q10 = np.percentile(tmp, 10)
        # print(self.q90, self.q10, self.q50, self.q75, self.q25)

    def diameter(self):
        furthest = csgraph.breadth_first_order(self.treeAdjmax, 0, directed=False, return_predecessors=False)[-1]
        dists = csgraph.shortest_path(self.treeAdjmax, return_predecessors=False, directed=False, indices=furthest)
        furfurthest = dists.argmax()
        diameter = dists[furfurthest]
        return diameter, furthest, furfurthest

    def stringy_value(self):
        diameter, _, _ = self.diameter()
        length = np.sum(self.treeAdjmax.toarray())

        return 100 * diameter / length

    def straight_value(self):
        diameter, a, b = self.diameter()
        dist = self.dist(a, b)
        return 100 * dist / diameter

    def outlying_value(self):
        w = self.q75 + 1.5 * (self.q75 - self.q25)
        cut_edges_length = 0
        tree_edges_length = 0
        for i in range(len(self.vertex[0])):
            tree_edges_length += self.treeAdjmax[self.vertex[0][i], self.vertex[1][i]]
            if self.treeAdjmax[self.vertex[0][i], self.vertex[1][i]] > w:
                 if np.sum(self.vertex == self.vertex[0][i]) == 1 or np.sum(self.vertex == self.vertex[1][i]) == 1:
                     cut_edges_length += self.treeAdjmax[self.vertex[0][i], self.vertex[1][i]]

        return 100 * (tree_edges_length - cut_edges_length) / tree_edges_length

    def skew_value(self):
        return 100 * (self.q90 - self.q50) / (self.q90 - self.q10)

    def striated_value(self):
        angles = {}
        for i in range(self.dot_num):
            if np.sum(self.vertex == i) == 2:
                angles[i] = []
        for j in range(len(self.vertex[0])):
            if self.vertex[0][j] in angles.keys():
                angles[self.vertex[0][j]].append(
                    (self.vertex[1][j], self.treeAdjmax[self.vertex[0][j], self.vertex[1][j]]))
            if self.vertex[1][j] in angles.keys():
                angles[self.vertex[1][j]].append(
                    (self.vertex[0][j], self.treeAdjmax[self.vertex[0][j], self.vertex[1][j]]))

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
        tmp_cut = -1
        tmpTreeAdjmax = copy.deepcopy(self.treeAdjmax).toarray()
        tmpTreeAdjmaxCut = np.zeros_like(tmpTreeAdjmax)
        for i in range(len(self.treeAdjmax.nonzero()[0])):
            idx = tmpTreeAdjmax.argmax()
            vi = idx // self.dot_num
            vj = idx % self.dot_num
            if tmpTreeAdjmax[vi][vj] > w:
                tmp_cut = tmpTreeAdjmax[vi][vj]
                tmpTreeAdjmax[vi][vj] = 0
                tmpTreeAdjmax[vj][vi] = 0

                _, labels = csgraph.connected_components(csgraph=csr_matrix(tmpTreeAdjmax + tmpTreeAdjmaxCut),
                                                         directed=False)
                labels_cnt = np.unique(labels)
                for j in range(np.max(labels) + 1):
                    labels_cnt[j] = np.sum(labels == j)
                # print(labels)
                if (max(0.05 * self.dot_num, 1) < labels_cnt).all() and (
                        labels_cnt < min(self.dot_num - 1, 0.95 * self.dot_num)).all():
                    tmpTreeAdjmaxCut[vi][vj] = 1
                    tmpTreeAdjmaxCut[vj][vi] = 1
                else:
                    break
            else:
                break
        if tmp_cut == -1:
            return 0
        else:
            return 100 * (1 - (tmp_cut - np.min(self.treeAdjmax.toarray())) /
                          (np.max(self.treeAdjmax.toarray() - np.min(self.treeAdjmax.toarray()))))

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
                if max(alpha * self.dot_num, 1) < subset_dots < min((1-alpha) * self.dot_num, self.dot_num-1):
                    max_after_cut = -i - 1
                else:
                    break
            else:
                break
        return 100 * (1 - self.sorted_treeEdge[max_after_cut][1] / self.sorted_treeEdge[-1][1])

def significance_outstanding1(data):
    if data.ndim != 1:
        return 0
    if data.dtype == "object":
        dataeleset = np.unique(data)
        data = np.array([int(np.argwhere(dataeleset == i))+1 for i in data])
    data = np.array(sorted(data))
    # maxv = data[0]
    # data = data / maxv
    idx = np.array([np.power(i, 0.7) for i in range(1, len(data) + 1)])
    k = np.sum((data - np.mean(data)) * (idx - np.mean(idx))) / np.sum((data - np.mean(data)) ** 2)
    b = np.mean(data) - k * np.mean(idx)
    data_pred = [k * i + b for i in idx]
    ssr = np.sum((data_pred - np.mean(data)) ** 2)
    sse = np.sum((data_pred - data) ** 2)

    f = ssr / (sse / len(data) - 2)
    return 100 * (1 - st.f.cdf(f, 1, len(data) - 2))


def significance_correlation2(data):
    if data.ndim != 2:
        return 0
    r = np.corrcoef(data[0], data[1])[0][1]
    n = data[0].shape[0]
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    t = np.abs(t)
    return 100 * (1 - 2 * (1 - st.t.cdf(t, n - 2)))


def significance_correlation(data):
    if data.ndim != 2:
        return 0
    if len(data) <= 2:
        return significance_correlation2(data)
    res = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            res.append(significance_correlation2(data[[i, j]]))
    return np.mean(res)

def significance_linearcorrelation(data):
    if data.ndim != 1:
        return 0
    if data.dtype == "object":
        dataeleset = np.unique(data)
        data = np.array([int(np.argwhere(dataeleset == i))+1 for i in data])
    x = [i for i in range(1, len(data) + 1)]
    k = np.sum((data - np.mean(data)) * (x- np.mean(x))) / np.sum((data - np.mean(data)) ** 2)
    b = np.mean(data) - k * np.mean(x)
    data_pred = [k * i + b for i in x]

    ssr = np.sum((data_pred - np.mean(data)) ** 2)
    sse = np.sum((data_pred - data) ** 2)

    f = ssr / (sse / len(data) - 2)
    return 100 * (1 - st.f.cdf(f, 1, len(data) - 2))


if __name__ == "__main__":
    data = np.random.rand(300, 2)
    label = np.random.randint(2, size=300)
    for i in range(300):
        data[i] = data[i] + 3 * label[i]

    # data = np.random.rand(200, 2)
    # data[49] = data[49] + 3
    # data = np.load("testdata/test1.npy")
    g = sciGraph(data)
    g.minSpanTree()
    print(g.outlying_value())
    print(g.skew_value())
    print(g.striated_value())
    print(g.stringy_value())
    print(g.straight_value())
    print(g.spearman_value())
    print(g.clumpy_value())
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.show()

    # data = np.array([10, 10, 10, 10, 100.2])
    #
    # print(significance_outstanding1(data))
    # print(significance_correlation(np.array([data, [5, 16, 22, 53, 10]])))
    # print(significance_correlation(data))
    # print(significance_linearcorrelation(data))

