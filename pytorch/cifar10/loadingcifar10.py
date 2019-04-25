# coding: utf-8
"""

"""
import torch
import torchvision
import torchvision.transforms as transforms


def get_set():
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='../../cifar10data', train=True, download=False, transform=transform)
    # 每次 loader 都 shuffle
    # 总共有5w张 train 图文，每次拿10个(batch_size=10)，能拿 5k 次(就是 main中循环的次数)，以此类推，
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=True, num_workers=2)

    return trainloader


if __name__ == "__main__":
    trainloader = get_set()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    n = 0

    for images, labels in trainloader:
        if n == 1:
            print(type(images))  # <class 'torch.Tensor'>
            print(type(images[0]))  # <class 'torch.Tensor'>
            print(images[0].size())  # torch.Size([3, 32, 32])

            # Tensor 转 Image
            img = transforms.ToPILImage()(images[0])
            print(type(img))  # <class 'PIL.Image.Image'>
            # img.show()

            # img = transforms.ToPILImage()(images[1])
            # img.show()
        n += 1

    print(n)  # 12500
