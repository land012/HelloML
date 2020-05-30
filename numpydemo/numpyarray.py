# coding: utf-8

import unittest
import numpy as np


class NumpyArrayTest(unittest.TestCase):

    @staticmethod
    def test1():
        arr1 = np.array([1, 2, 3, 1, 2, 1])
        print(type(arr1))  # <class 'numpy.ndarray'>
        print(arr1)

    @staticmethod
    def test2():
        arr2 = np.array(["a", "b", "c"])
        print(arr2)
        print(arr2[0:2])  # ['a' 'b']

    @staticmethod
    def test_zeros():
        """ 逐个元素计算 """
        arr1 = np.zeros(3)
        print(type(arr1))  # <class 'numpy.ndarray'>
        print(arr1)  # [0. 0. 0.]

        arr2 = arr1 + np.array([1, 2, 3])
        arr3 = arr2 + np.array([4, 5, 6])
        print(arr2)  # [1. 2. 3.]
        print(arr3)  # [5. 7. 9.]

        arr4 = arr3 / 2
        print(arr4)  # [2.5 3.5 4.5]

    @staticmethod
    def test_arrcal():
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        arr3 = arr1 * arr2
        print(arr3)  # [ 4 10 18]
        print(sum(arr3))  # 32

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(NumpyArrayTest('test_zeros'))
    suite.addTest(NumpyArrayTest('test_arrcal'))

    unittest.TextTestRunner(verbosity=2).run(suite)
