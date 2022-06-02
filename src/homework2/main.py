import pandas as pd
import numpy as np
import Kmeans
import  visualization

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import  SpectralClustering
from sklearn.metrics.cluster import pair_confusion_matrix


def read_data_csv():
    df = pd.read_csv('../../dataset/cluster.csv')
    x = df.values[:, 1:-1]
    label = df.values[:, -1:]
    label_set = np.unique(label)
    return x, label, label_set.shape


def calc_purity(true, predict, n_cluster):

    total = 0
    for i in range(n_cluster):
        _, counts = np.unique(true[np.where(predict == i)], return_counts=True)
        if counts.shape[0] != 0:
            total += np.max(counts)
    return total / true.shape[0]

def calc_rand_index(true, predict):
    (tn, fp), (fn, tp) = pair_confusion_matrix(true, predict)
    return (tp + tn) / (tp + tn + fp + fn)


def model_predict(model, x, label, n_cluster, type):
    label = np.squeeze(label, axis=1)
    out_label = model.fit_predict(x)
    visualization.draw_scatter(x, "type", out_label)
    purity = calc_purity(label, out_label, n_cluster)
    ri = calc_rand_index(label, out_label)

    print(type + "_purity:" + str(purity))
    print(type + "_randIndex:" + str(ri))
    return purity, ri


if __name__ == "__main__":
    x, label, _ = read_data_csv()

    algorithms = ["kmeans", "spectral", "dbscan"]
    for algorithm in algorithms:
        if algorithm == "kmeans":
            choices = ["sklearn", "our"]
            n_clusters = [3, 6, 12, 24, 100]
            for choice in choices:
                if choice == "sklearn":
                    res = []
                    for n_cluster in n_clusters:
                        model = KMeans(n_clusters=n_cluster, max_iter=100)
                        purity, ri = model_predict(model, x, label, n_cluster,
                                      algorithm + "_K_" + str(n_cluster) + "_" + choice)
                        res.append([purity, ri])
                else:
                    res_our = []
                    for n_cluster in n_clusters:
                        model = KMeans(n_clusters=n_cluster, max_iter=100)
                        purity, ri = model_predict(model, x, label, n_cluster,
                                                   algorithm + "_K_" + str(n_cluster) + "_" + choice)
                        res_our.append([purity, ri])
            visualization.draw_line_map(np.vstack((np.array(res).T, np.array(res_our).T)),
                                        "Kmeans", n_clusters, ["purity", "RI", "our_purity", "our_RI"])
        elif algorithm == "DBSCAN":
            model = DBSCAN()
            model_predict(model, x, label, 12, algorithm)
        else:
            model = SpectralClustering()
            model_predict(model, x, label, 12, algorithm)
