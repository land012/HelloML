# coding: utf-8
import torch
import unittest


class TensorTest(unittest.TestCase):

    @staticmethod
    def test1():
        x = torch.Tensor([1, 2, 3])
        print(x)
        print(x.size())  # torch.Size([3])

        x.requires_grad_(True)

        y = x.sum()
        print(y)  # tensor(6., grad_fn=<SumBackward0>)
        print(y.grad_fn)

        y.backward()
        print(x.grad)  # tensor([1., 1., 1.])

        z = x.mean()
        print(z)  # tensor(2., grad_fn=<MeanBackward1>)
        z.backward()
        print(x.grad)  # tensor([1.3333, 1.3333, 1.3333])
