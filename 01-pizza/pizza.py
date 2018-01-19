from sklearn import datasets

import pandas as pandas

input_file = "pizzas.csv"
pizzas = pandas.read_csv(input_file, header = 0)
print(pizzas.keys())
