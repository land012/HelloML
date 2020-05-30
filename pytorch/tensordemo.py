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

    @staticmethod
    def test2():
        """
        view
        :return:
        """
        t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        t1 = t1.view(6)
        print(t1)  # tensor([1, 2, 3, 4, 5, 6])

        t1 = t1.view(3, 2)
        """
        tensor([[1, 2],
                [3, 4],
                [5, 6]])
        """
        print(t1)

    @staticmethod
    def test3():
        t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])

        t1 = t1.view(6)

        # 第1维 取决于第2维，
        # 因为第2维是3，所以第1维是2
        t2 = t1.view(-1, 3)
        """
        tensor([[1, 2, 3],
                [4, 5, 6]])
        """
        print(t2)

        t3 = t1.view(-1, 2)
        """
        tensor([[1, 2],
                [3, 4],
                [5, 6]])
        """
        print(t3)

    @staticmethod
    def test4():
        t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])

        # 第1维 取决于第2维，
        # 因为第2维是2，所以第1维是3
        t2 = t1.view(-1, 2)
        """
        tensor([[1, 2, 3],
                [4, 5, 6]])
        """
        print(t2)

