# coding: utf-8
"""

"""
import torch
import torchvision
import numpy as np


if __name__ == "__main__":
    trainset = torchvision.datasets.CIFAR10(root='../../cifar10data', train=True, download=False, transform=None)
    img_tensor = trainset.__getitem__(0)
    print(type(img_tensor))  # <class 'tuple'>
    print(len(img_tensor))  # 2
    image = img_tensor[0]
    label = img_tensor[1]
    print(type(image))  # <class 'PIL.Image.Image'>
    print(image.height, image.width)  # 32 32
    print(type(label))  # <class 'int'>

    # Image 转 Tensor
    img_tensor = torchvision.transforms.ToTensor()(image)  # C x H x W
    print(type(img_tensor))  # <class 'torch.Tensor'>
    print(img_tensor.size())  # torch.Size([3, 32, 32]) # C x H x W RGB?
    """
    tensor([[[0.2314, 0.1686, 0.1961,  ..., 0.6196, 0.5961, 0.5804],
             [0.0627, 0.0000, 0.0706,  ..., 0.4824, 0.4667, 0.4784],
             [0.0980, 0.0627, 0.1922,  ..., 0.4627, 0.4706, 0.4275],
             ...,
             [0.8157, 0.7882, 0.7765,  ..., 0.6275, 0.2196, 0.2078],
             [0.7059, 0.6784, 0.7294,  ..., 0.7216, 0.3804, 0.3255],
             [0.6941, 0.6588, 0.7020,  ..., 0.8471, 0.5922, 0.4824]],

            [[0.2431, 0.1804, 0.1882,  ..., 0.5176, 0.4902, 0.4863],
             [0.0784, 0.0000, 0.0314,  ..., 0.3451, 0.3255, 0.3412],
             [0.0941, 0.0275, 0.1059,  ..., 0.3294, 0.3294, 0.2863],
             ...,
             [0.6667, 0.6000, 0.6314,  ..., 0.5216, 0.1216, 0.1333],
             [0.5451, 0.4824, 0.5647,  ..., 0.5804, 0.2431, 0.2078],
             [0.5647, 0.5059, 0.5569,  ..., 0.7216, 0.4627, 0.3608]],

            [[0.2471, 0.1765, 0.1686,  ..., 0.4235, 0.4000, 0.4039],
             [0.0784, 0.0000, 0.0000,  ..., 0.2157, 0.1961, 0.2235],
             [0.0824, 0.0000, 0.0314,  ..., 0.1961, 0.1961, 0.1647],
             ...,
             [0.3765, 0.1333, 0.1020,  ..., 0.2745, 0.0275, 0.0784],
             [0.3765, 0.1647, 0.1176,  ..., 0.3686, 0.1333, 0.1333],
             [0.4549, 0.3686, 0.3412,  ..., 0.5490, 0.3294, 0.2824]]])
    """
    print(img_tensor)

    img_ndarray = img_tensor.numpy()
    img_ndarray = np.transpose(img_ndarray, (1, 2, 0))
    print(type(img_ndarray))  # <class 'numpy.ndarray'>
