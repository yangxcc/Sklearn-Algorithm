import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

data = sns.load_dataset("iris")
data.head()
sns.pairplot(data, hue='species')
plt.show()

def trainModel(data, clusterNum):
    """
    使用KMeans对数据进行聚类
    """
    # max_iter表示EM算法迭代次数，n_init表示K-means算法迭代次数，algorithm="full"表示使用EM算法。
    model = KMeans(n_clusters=clusterNum, max_iter=100, n_init=10, algorithm="full")
    model.fit(data)
    return model


def computeSSE(model, data):
    """
    计算聚类结果的误差平方和
    """
    wdist = model.transform(data).min(axis=1)
    sse = np.sum(wdist ** 2)
    return sse

if __name__ == "__main__":
    col = [['petal_width', 'sepal_length'], ['petal_width', 'petal_length'], ['petal_width', 'sepal_width'], ['sepal_length', 'petal_length'],
['sepal_length', 'sepal_width'], ['petal_length', 'sepal_width']]

    for i in range(6):
        fig = plt.figure(figsize=(8, 8), dpi=80)
        ax = fig.add_subplot(3, 2, i+1)
        sse = []
        for j in range(2, 6):
            model = trainModel(data[col[i]], j)
            sse.append(computeSSE(model, data[col[i]]))
        ax.plot(range(2,6), sse, 'k--', marker="o",markerfacecolor="r", markeredgecolor="k")
        ax.set_xticks([1,2,3,4,5,6])
        title = "clusterNum of %s and %s" % (col[i][0], col[i][1])
        ax.title.set_text(title)
        plt.show()

    petal_data = data[['petal_width', 'petal_length']]
    model = trainModel(petal_data, 3)
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    colors = ["r", "b", "g"]
    ax.scatter(petal_data.petal_width, petal_data.petal_length, c=[colors[i] for i in model.labels_],
               marker="o", alpha=0.8)
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker="*", c=colors, edgecolors="white",
               s=700., linewidths=2)
    yLen = petal_data.petal_length.max() - petal_data.petal_length.min()
    xLen = petal_data.petal_width.max() - petal_data.petal_width.min()
    lens = max(yLen + 1, xLen + 1) / 2.
    ax.set_xlim(petal_data.petal_width.mean() - lens, petal_data.petal_width.mean() + lens)
    ax.set_ylim(petal_data.petal_length.mean() - lens, petal_data.petal_length.mean() + lens)
    ax.set_ylabel("petal_length")
    ax.set_xlabel("petal_width")
    plt.show()


