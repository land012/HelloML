# coding: utf-8
"""
# Created by xudazhou at 2019/4/23
逻辑回归
"""
import numpy as np


def load_dataset(f_name):
    """load dataset"""
    dataMat = []
    labelMat = []
    f = open(f_name)
    for line in f:
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    f.close()
    return dataMat, labelMat


def sigmoid(x_mat):
    """预测函数"""
    return 1.0 / (1 + np.exp(-x_mat))


def grad_descent(p_data_arr, p_label_arr):
    """梯度下降计算参数"""
    l_data_mat = np.mat(p_data_arr)
    l_label_mat = np.mat(p_label_arr).transpose()
    m, n = np.shape(l_data_mat)
    alpha = 0.001
    max_cycles = 10
    weights = np.ones((n, 1))

    for k in range(max_cycles):
        h = sigmoid(l_data_mat * weights)
        error = h - l_label_mat
        temp = l_data_mat.transpose() * error
        weights -= alpha * temp

    return weights


if __name__ == "__main__":
    data_mat, label_mat = load_dataset("trainingSample.txt")
    mat_a = grad_descent(data_mat, label_mat)
    print("================================= weights")
    print(mat_a)  # 3 x 1

    data_test, label_test = load_dataset("testingSample.txt")
    mat_h = sigmoid(np.mat(data_test) * mat_a)
    print("================================== predictor")
    print(mat_h.transpose())

    print("================================== error")
    mat1 = np.mat([np.array(mat_h.transpose())[0], label_test]).transpose()
    print(mat1)
