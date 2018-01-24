from sklearn import datasets
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data

model = KMeans(n_clusters=5)
model.fit(X)

centroids = model.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="^", s=200)
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.show()
