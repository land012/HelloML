# coding: utf-8
import torch
from torch.autograd import Variable

a = Variable(torch.ones(2, 2), requires_grad=True)
"""
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
"""
print(a)

b = a + 1
"""
tensor([[2., 2.],
        [2., 2.]], grad_fn=<AddBackward0>)
"""
print(b)

"""
tensor([[2., 2.],
        [2., 2.]])
"""
print(b.data)
print(b.grad_fn)  # <AddBackward0 object at 0x0000000002937710>
out = b.mean()
print(out)  # tensor(2., grad_fn=<MeanBackward1>)

out.backward()
"""
tensor([[0.2500, 0.2500],
        [0.2500, 0.2500]])
"""
print(a.grad)
