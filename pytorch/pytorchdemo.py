# coding: utf-8

import torch
import unittest


class PytorchDemo(unittest.TestCase):

    @staticmethod
    def test1():
        x = torch.rand(3, 3)
        """
        tensor([[0.0407, 0.7449, 0.3779],
                [0.8378, 0.9546, 0.3357],
                [0.8776, 0.4826, 0.3453]])
        """
        print("tensor1:" + str(x))

        """
        tensor([0.3779, 0.3357, 0.3453])
        """
        print(x[:, 2])
        """
        tensor([[0.3779],
            [0.3357],
            [0.3453]])
        """
        print(x[:, 2:3])
        print(x.grad_fn)  # None

        x += 1
        print(x.grad_fn)  # None

        x.requires_grad_(True)
        # ，这样会报错 RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
        # x += 1
        y = x + 1
        print(y.grad_fn)  # <AddBackward0 object at 0x00000000028AB128>

    @staticmethod
    def test2():
        x = torch.rand(3, 3)
        """
        tensor([[0.5484, 0.0385, 0.7197],
                [0.1494, 0.9357, 0.3055],
                [0.2294, 0.1096, 0.2362]])
        """
        print(x)
        print(x[0, 2])  # tensor(0.7197) 0行2列
        print(x[1, 0])  # tensor(0.1494) 1行0列
        print(torch.max(x))  # tensor(0.9357)
        print(torch.max(x, 0))  # (tensor([0.5484, 0.9357, 0.7197]), tensor([0, 1, 0])) 每列的最大值
        print(torch.max(x, 1))  # (tensor([0.7197, 0.9357, 0.2362]), tensor([2, 1, 2])) 每行的最大值

    @staticmethod
    def test3():
        x = torch.rand(3, 3, 3)
        """
        tensor([[[0.1787, 0.3517, 0.9177],
                 [0.5419, 0.0459, 0.5001],
                 [0.6968, 0.3446, 0.4121]],

                [[0.7568, 0.7669, 0.8229],
                 [0.0556, 0.5934, 0.3537],
                 [0.9668, 0.7497, 0.2314]],

                [[0.6984, 0.5804, 0.7193],
                 [0.2365, 0.3349, 0.7648],
                 [0.2047, 0.2567, 0.3011]]])
        """
        print(x)
        print(x[0, 2, 1])  # tensor(0.3446)
        print(x[1, 0, 2])  # tensor(0.8229)
        print(torch.max(x))  # tensor(0.9668)
        """
        (tensor([[0.7568, 0.7669, 0.9177],
                [0.5419, 0.5934, 0.7648],
                [0.9668, 0.7497, 0.4121]]),
        tensor([[1, 1, 0],
                [0, 1, 2],
                [1, 1, 0]]))
        """
        print(torch.max(x, 0))
        """
        (tensor([[0.6968, 0.3517, 0.9177],
                [0.9668, 0.7669, 0.8229],
                [0.6984, 0.5804, 0.7648]]),
        tensor([[2, 0, 0],
                [2, 0, 0],
                [0, 0, 1]]))
        """
        print(torch.max(x, 1))
