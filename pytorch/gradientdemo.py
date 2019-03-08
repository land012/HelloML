# coding: utf-8
import torch

if __name__ == "__main__":
    x = torch.Tensor([1, 2, 3])
    print(x)
    print(x.size())  # torch.Size([3])

    x.requires_grad_(True)

    z = x.mean()
    print(z)  # tensor(2., grad_fn=<MeanBackward1>)
    z.backward()
    print(x.grad)  # tensor([0.3333, 0.3333, 0.3333])
