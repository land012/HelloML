# coding: utf-8
"""
# Created by xudazhou at 2019/3/7
卷积神经网络
"""
import loadingcifar10
import lenetdemo2
import torch
import torch.optim as optim
import torch.nn as nn


def main():
    net = lenetdemo2.Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainloader, testloader = loadingcifar10.get_set()

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:
                print("%d %5d %.3f" % (epoch, i + 1, running_loss / 2000))
                running_loss = 0.0

    testiter = iter(testloader)
    test_imgs, test_labels = testiter.next()
    for i in test_labels:
        print("test_label %d" % i)  # 6 1 0 0 6

    test_outputs = net(test_imgs)
    """
    test_outputs:tensor([[-1.0252e+00, -8.3855e-01,  7.8499e-01,  7.6871e-01,  1.7534e-02,
              2.9006e-02,  3.1471e+00, -3.2825e-01, -2.4518e+00,  3.0189e-01],
            [ 4.1685e+00,  4.7839e+00, -3.7711e+00, -1.9065e+00, -3.4416e+00,
             -3.9533e+00, -2.7144e+00, -4.9113e+00,  6.4453e+00,  4.8884e+00],
            [ 7.5426e+00,  2.9692e+00,  7.7428e-05, -3.3238e+00, -9.1390e-01,
             -5.8448e+00, -3.5148e+00, -3.9876e+00,  4.9755e+00,  2.6781e+00],
            [ 5.7542e+00, -9.3433e-01,  1.9797e+00, -1.5910e+00,  4.9292e-01,
             -3.4809e+00, -1.8062e+00, -4.1670e+00,  4.1703e+00, -5.0784e-01],
            [ 1.0443e+00,  2.4668e+00, -2.0657e+00, -2.9487e-01, -2.1651e+00,
             -1.7653e+00,  5.1260e-01, -2.9273e+00,  2.2977e+00,  3.4849e+00]],
           grad_fn=<AddmmBackward>)
    """
    print("test_outputs:" + str(test_outputs))
    max_dim, predicteds = torch.max(test_outputs, 1)
    print("max_dim:" + str(max_dim))  # 最大值 tensor([3.1471, 6.4453, 7.5426, 5.7542, 3.4849], grad_fn=<MaxBackward0>)
    print("predicteds type:" + str(type(predicteds)))  # predicteds type:<class 'torch.Tensor'>
    print("predicteds:" + str(predicteds))  # predicteds:tensor([6, 8, 0, 0, 9])


if __name__ == "__main__":
    main()
