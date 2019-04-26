# coding: utf-8
import torch
import unittest


class GradDemo(unittest.TestCase):

    @staticmethod
    def test1():
        x = torch.Tensor([1, 2, 3])
        print(x)  # tensor([1., 2., 3.])
        print(x.dtype)  # torch.float32
        print(x.size())  # torch.Size([3])

        x.requires_grad_(True)

        z = x.mean()
        print(z)  # tensor(2., grad_fn=<MeanBackward1>)
        z.backward()
        print(x.grad)  # tensor([0.3333, 0.3333, 0.3333])

    @staticmethod
    def test2():
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float, requires_grad=True)
        print(x)
        print(x.size())  # torch.Size([2, 3])

        y = x.mean()
        print(y)  # tensor(3.5000, grad_fn=<MeanBackward1>)

        y.backward()
        """
        tensor([[0.1667, 0.1667, 0.1667],
                [0.1667, 0.1667, 0.1667]])
        """
        print(x.grad)

    @staticmethod
    def test3():
        x = torch.ones(2, 2, requires_grad=True)
        print(x)

        y = x + 2

        z = y * y * 3
        z.backward(torch.ones(2, 2))
        """
        tensor([[18., 18.],
                [18., 18.]])
        """
        print(x.grad)
