# coding: utf-8
"""
# Created by xudazhou at 2019/12/16
"""
import tensorflow as tf
import numpy

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data(r"D:\_ml_data\mnist.npz")
print(numpy.shape(train_images))  # (60000, 28, 28)
train_images = train_images / 255.0
print(numpy.shape(train_images))  # (60000, 28, 28)

print(numpy.shape(test_images))
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)

print(model.evaluate(test_images, test_labels))  # [0.10696243395525962, 0.966]
print(model.metrics_names)  # ['loss', 'acc']

print(model.predict_classes(test_images[0:4]))  # [7 2 1 0]
print(test_labels[0:4])  # [7 2 1 0]
