# coding: utf-8
"""
# Created by xudazhou at 2019/12/16
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data(r"D:\_ml_data\mnist.npz")
print(type(train_images))  # <class 'numpy.ndarray'>
print(numpy.shape(train_images))  # (60000, 28, 28)
print(len(train_labels))  # 60000

# 显示图片
plt.imshow(train_images[2])
plt.show()
