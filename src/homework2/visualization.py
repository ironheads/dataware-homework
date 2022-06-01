import os
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

def draw_line_map(data, title, x, labels):
    plt.clf()
    shape = data.shape
    if len(labels) != shape[0]:
        print("label size wrong")
        return
    data = data.reshape(shape[0] * shape[1])
    pd_label = []
    for label in labels:
        pd_label += [label] * shape[1]
    x = x * shape[0]
    df = pd.DataFrame(dict(clusters=x, value=data, type=pd_label))
    sns.lineplot(data=df, x='clusters', y='value', hue="type")
    plt.title(title)
    plt.show()

def draw_scatter(data, title, labels):
    plt.clf()
    shape = data.shape[0]
    est = KBinsDiscretizer(n_bins=[100, 100], encode='ordinal').fit(data)
    data = est.transform(data)
    if len(labels) != shape:
        print("label size wrong")
        return
    df = pd.DataFrame(dict(x=data[:, 0], y=data[:, 1], type=labels))
    sns.scatterplot(data=df, x='x', y='y', hue='type', s=2, palette="deep")
    plt.title(title)
    plt.show()