import pandas as pandas

input_file = "bananas.csv"
dfBananas = pandas.read_csv(input_file, header = 0)
X = dfBananas.iloc[:, :-1]
y = dfBananas.iloc[:, -1]

print("X", X)
print("y", y)
