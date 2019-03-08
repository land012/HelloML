from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
digits = datasets.load_digits()

print(type(iris))  # <class 'sklearn.utils.Bunch'>
print(type(digits))  # <class 'sklearn.utils.Bunch'>

data = digits.data
target = digits.target
print(type(data))  # <class 'numpy.ndarray'>
print(np.shape(data))  # (1797, 64)
print(type(target))
print(np.shape(target))
