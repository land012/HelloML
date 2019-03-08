# coding: utf-8

import torch

x = torch.ones(5)
print(x)

y = x.numpy()
print(y)  # [1. 1. 1. 1. 1.]

z = x + 1
print(x)  # tensor([1., 1., 1., 1., 1.])
print(z)  # tensor([2., 2., 2., 2., 2.])
print(z.grad_fn)

z.add_(1)
print(z)
