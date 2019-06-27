#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: cnn_tensorflow.py
@time: 2019/05/03
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

## strides [1,a,b,1] --- a表示x轴移动的步长，b 表示y轴移动的步长
def conv2d(x, w):
    # 由于步长为１　并且ｐａｄｄｉｎｇ的模式是ｓａｍｅ　所以不改变原图像的尺寸
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

# 将接受到的卷积层的数据，按照一定的参数范围，池化成一个点（２×２池化成一个点）
def max_pool_2X2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784])
    #最后一个１表示通道
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y_true = tf.placeholder(tf.float32, [None, 10])
    # 最后一个１表示通道数，32表示卷积核的数量
    w_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    # 第一卷积 28*28
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    # 第一池化 14*14
    h_pool1 = max_pool_2X2(h_conv1)
    # 开始第二次卷积

    # 最后一个64表示卷积核的数量
    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    # 第二卷积  14*14
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    # 第二池化 7*7
    h_pool2 = max_pool_2X2(h_conv2)

    # 全连接层
    w_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    # 对第二卷积层的输出进行变形
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    # 加一个隐层
    hidden1 =  tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob=keep_prob)

    # 最后一个全连接层
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(hidden1_drop, w_fc2)+b_fc2)


    # loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y), reduction_indices=[1]))
    # optimizor
    opt = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # init
    tf.global_variables_initializer().run()
    # TRAINNING
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        if i%100==0:
            train_accu = accuracy.eval({x: batch_xs, y_true: batch_ys, keep_prob: 1})
            print('step %d, train_accuracy %g'%(i, train_accu))
        opt.run({
            x: batch_xs, y_true: batch_ys, keep_prob: 0.5
        })
    # PREDICTING

    print(accuracy.eval({x: mnist.test.images, y_true: mnist.test.labels, keep_prob: 1}))


