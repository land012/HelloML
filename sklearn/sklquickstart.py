from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()

print(type(iris))
print(type(digits))

print(digits.data)
