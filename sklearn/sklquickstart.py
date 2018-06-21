from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
digits = datasets.load_digits()

print(type(iris))
print(type(digits))

data = digits.data
target = digits.target
print(type(data))
print(np.shape(data))
print(type(target))
print(np.shape(target))
