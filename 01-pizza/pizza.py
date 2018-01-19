from sklearn import datasets

import pandas as pandas

input_file = "pizzas.csv"
pizzas = pandas.read_csv(input_file, header = 0)

print("Start linear regression with pizzas\n")
print("Example of data with the first 3 lines:")
print(pizzas[:3])
