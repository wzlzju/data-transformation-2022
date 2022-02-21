import numpy as np


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


if __name__ == "__main__":
    for i in range(10):
        data = np.random.rand(1000, 2)
        label = np.random.randint(10, size=1000)
        for i in range(1000):
            data[i] = data[i] + label[i]*10000
        print(CDM(data, label))
