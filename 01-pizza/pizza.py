import warnings
warnings.filterwarnings(action='ignore', module='scipy', message='^internal gelsd')

import pandas as pandas
from sklearn.linear_model import LinearRegression

input_file = 'pizzas.csv'
dfPizzas = pandas.read_csv(input_file, header = 0)

print('Start linear regression with pizzas\n')
print('Example of data with the first 3 lines:')
print(dfPizzas[:3], end='\n\n')

X = dfPizzas.iloc[:, :-1]
y = dfPizzas.iloc[:, -1]

model = LinearRegression()
model.fit(X, y)

diametreEssai = 18

prediction = model.predict([[diametreEssai]])[0]

print('Prediction avec un diametre', diametreEssai, '-->', prediction)
