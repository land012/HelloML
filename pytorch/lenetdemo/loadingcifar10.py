# coding: utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def get_set():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../../cifar10data', train=True, download=False, transform=transform)
    # 每次 loader 都 shuffle
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../../cifar10data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=True, num_workers=2)

    return trainloader, testloader


if __name__ == "__main__":
    trainloader, testloader = get_set()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    img = torchvision.utils.make_grid(images) / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

    for i in labels:
        print("%d %s" % (i, classes[i]))
