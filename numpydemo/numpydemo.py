# coding: utf-8
"""
# Created by xudazhou at 2019/4/25
"""

import numpy as np
import unittest
# import numpy.nprandom


class NumpyDemo(unittest.TestCase):

    @staticmethod
    def test1():
        print(np.__version__)  # 1.16.1

        arr1 = np.array([[[1, 2], [3, 4]],
                         [[5, 6], [7, 8]],
                         [[9, 10], [11, 12]]])
        """
        [[[ 1  2] [ 3  4]]
         [[ 5  6] [ 7  8]]
         [[ 9 10] [11 12]]]
        """
        print(arr1)
        print(np.shape(arr1))  # (3, 2, 2)
        """
        [[[1 2]
          [3 4]]]
        """
        print(arr1[0:1])

        arr2 = np.transpose(arr1, (1, 2, 0))
        """
        [[[ 1  5  9]
          [ 2  6 10]]
         [[ 3  7 11]
          [ 4  8 12]]]
        """
        print(arr2)

    # @staticmethod
    # def test2():
    #     list1 = numpy.nprandom.randint(3, size=2)
