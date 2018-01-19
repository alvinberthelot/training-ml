import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import matplotlib.pyplot as plt

boston = datasets.load_boston()

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("R squared :", model.score(X_test, y_test))
print("Mean squared error :", metrics.mean_squared_error(y_test, predictions))

plt.scatter(y_test, predictions)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.show()



