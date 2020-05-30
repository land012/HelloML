# coding: utf-8
"""
# Created by xudazhou at 2020/1/3
线性回归
"""
import paddle

paddle.dataset.uci_housing.load_data("D:\\_ml_data\\housing.data")

# 会从 git 下载，无法指定本地路径
print(type(paddle.dataset.uci_housing.train()))  # <class 'function'>
