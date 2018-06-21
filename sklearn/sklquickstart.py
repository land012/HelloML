from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()

print(type(iris))
print(type(digits))

data = digits.data
target = digits.target
print(type(data))
print(type(target))
