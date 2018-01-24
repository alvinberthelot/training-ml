from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import metrics

import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
# plt.scatter(X_train[:, 2], X_train[:, 1], c=y_train)
# plt.show()

model = neighbors.KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy score :", metrics.accuracy_score(y_test, predictions))
print("Classification report :\n", metrics.classification_report(y_test, predictions))
print("Confusion matrix :\n", metrics.confusion_matrix(y_test, predictions))
