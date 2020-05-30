# coding: utf-8

import numpy as np
import unittest


class NumpyDemo(unittest.TestCase):

    @staticmethod
    def test1():
        """均值 方差"""
        arr1 = np.array([1, 2, 3, 4, 5])
        print(np.mean(arr1))  # 3.0
        print(np.var(arr1))  # 2.0

        arr2 = np.array([2, 2, 3, 4, 4])
        print(np.var(arr2))  # 0.8

    @staticmethod
    def test2():
        """下标从 0 开始"""
        arr1 = np.array([4, 3, 6, 2])
        print(arr1[0])  # 4

        arr1.append(3)
        print(arr1)

    @staticmethod
    def test3():
        """
        90分位
        变异值对结果有较大影响
        """
        arr1 = np.array([1, 2, 3, 4, 5])
        print(np.percentile(arr1, 90))  # 4.6
        print(np.percentile(arr1, 90, interpolation="nearest"))  # 5
        print(np.percentile(arr1, 80))  # 4.2
        print(np.percentile(arr1, 80, interpolation="nearest"))  # 4

        arr2 = np.array([1, 2, 3, 4, 10])
        print(np.percentile(arr2, 90))  # 7.6
        print(np.percentile(arr2, 90, interpolation="nearest"))  # 10
        print(np.percentile(arr2, 80))  # 5.2
        print(np.percentile(arr2, 80, interpolation="nearest"))  # 4

    @staticmethod
    def test4():
        """对数组 slice"""
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = arr1[:, 2]
        print(arr2)  # [3 6]

        list2 = [ line[0] for line in arr1 ]
        print(list2)  # [1, 4]

    @staticmethod
    def test_max():
        arr1 = np.array([1, 2, 3, 4])
        print(arr1.max())  # 4
        print(np.max(arr1))  # 4

    @staticmethod
    def test_unique():
        """
        排序并统计出现次数
        """
        arr1 = np.array(['e', 'b', 'e', 'a', 'd', 'd'])
        arr11, arr12 = np.unique(arr1, return_counts=True)
        print(arr11)  # ['a' 'b' 'd' 'e']
        print(arr12)  # [1 1 2 2]

    @staticmethod
    def test_unique2():
        """
        获取出现次数最多的元素
        """
        arr1 = np.array(['e', 'b', 'e', 'a', 'd', 'd'])
        arr11, arr12 = np.unique(arr1, return_counts=True)
        print(arr11)  # ['a' 'b' 'd' 'e']
        print(arr12)  # [1 1 2 2]
        index1 = np.argmax(arr12)
        print(index1)  # 2
        print(arr11[index1])  # d

    @staticmethod
    def test_bincount():
        """统计出现次数，不存在的数字补0"""
        arr1 = np.array([4, 4, 4, 9, 20])
        b1 = np.bincount(arr1)
        print(b1)  # [0 0 0 0 3 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1]

    @staticmethod
    def test_argmax():
        """打印最大数值的下标"""
        arr1 = np.array([4, 9, 4, 9, 3])
        am1 = np.argmax(arr1)
        print(type(am1))  # <class 'numpy.int64'>
        print(am1)  # 1
        # arr2 = np.array([])
        # print(np.argmax(arr1, 0, arr2))
        # print(arr2)

    @staticmethod
    def test_argmax2():
        """打印最大数值的下标"""
        arr1 = np.array([[4, 9, 4], [9, 3, 10]])
        print(arr1)

        am1 = np.argmax(arr1)
        print(type(am1))  # <class 'numpy.int64'>
        print(am1)  # 5

        am1 = np.argmax(arr1, 1)
        print(type(am1))  # <class 'numpy.ndarray'>
        print(am1)  # [1 2]

    @staticmethod
    def test_ones():
        arr = np.ones((3, 2))
        print(type(arr))  # <class 'numpy.ndarray'>
        """
        [[1. 1.]
         [1. 1.]
         [1. 1.]]
        """
        print(arr)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(NumpyDemo('test4'))

    unittest.TextTestRunner(verbosity=0).run(suite)
