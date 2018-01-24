from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

iris = datasets.load_iris()

# print(iris.keys())
# print(iris.target_names)
# print(iris.feature_names)
# print(iris.data[:3])
# print(iris.target[:3])

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# kernel : rbf, sigmoid, linear, poly
model = SVC(kernel="linear")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy score :", metrics.accuracy_score(y_test, predictions))
print("Classification report :\n", metrics.classification_report(y_test, predictions))
print("Confusion matrix :\n", metrics.confusion_matrix(y_test, predictions))