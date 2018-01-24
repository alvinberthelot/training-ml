from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import metrics

import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
# plt.scatter(X_train[:, 2], X_train[:, 1], c=y_train)
plt.show()