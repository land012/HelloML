# coding: utf-8
"""
用于手写体字符识别的非常高效的卷积神经网络
"""
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 卷积 3个input_channels，6个卷积核，所以6个output_channel，卷积核大小 5x5
        # 3 个 input channel，6 个 filter，是怎么输出 6 个 output channel 的？
        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)  # 池化 采样窗口 2x2，步长2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 400 - 120
        self.fc2 = nn.Linear(120, 84)  # 120 - 84
        self.fc3 = nn.Linear(84, 10)  # 84 - 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积 6x28x28 激励 池化 6x14x14
        x = self.pool(F.relu(self.conv2(x)))  # 卷积 16x10x10 激励 池化 16x5x5
        x = x.view(-1, 16 * 5 * 5)  # 改变维度 1 x 400
        x = F.relu(self.fc1(x))  # 线性 激活 输出 1 x 84
        x = F.relu(self.fc2(x))  # 线性 激活 输出 1 x 10
        x = self.fc3(x)
        return x
