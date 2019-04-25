# coding: utf-8

import torch
import unittest


class OnesTest(unittest.TestCase):

    @staticmethod
    def test1():
        x = torch.ones(5)
        print(x)  # tensor([1., 1., 1., 1., 1.])

        y = x.numpy()
        print(y)  # [1. 1. 1. 1. 1.]

        z = x + 1
        print(x)  # tensor([1., 1., 1., 1., 1.])
        print(z)  # tensor([2., 2., 2., 2., 2.])
        print(z.grad_fn)

        z.add_(1)
        print(z)

    @staticmethod
    def test2():
        t1 = torch.ones(3, 2)
        """
        tensor([[1., 1.],
                [1., 1.],
                [1., 1.]])
        """
        print(t1)
