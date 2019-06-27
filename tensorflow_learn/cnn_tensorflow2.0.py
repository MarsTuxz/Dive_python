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
'''
升级班的ｃｎｎ

'''

def weight_variable_l2(shape, stddev, wl):
    '''

    :param shape:
    :param stddev:标准差
    :param wl: l2 loss的大小
    :return:
    '''
    initial = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(initial), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return initial

def bias_variable(shape, value=0.1):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial)

## strides [1,a,b,1] --- a表示x轴移动的步长，b 表示y轴移动的步长
def conv2d(x, w):
    # 由于步长为１　并且ｐａｄｄｉｎｇ的模式是ｓａｍｅ　所以不改变原图像的尺寸
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

# 将接受到的卷积层的数据，按照一定的参数范围，池化成一个点（２×２池化成一个点）
def max_pool_2X2(x):
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

if __name__ == '__main__':
    batch_size = 50

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784])
    #最后一个１表示通道
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y_true = tf.placeholder(tf.float32, [None, 10])
    # 最后一个１表示通道数，32表示卷积核的数量
    w_conv1 = weight_variable_l2([5,5,1,32], stddev=5e-2, wl=0.0)
    b_conv1 = bias_variable([32])
    # 第一卷积 28*28
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    # 第一池化 14*14
    h_pool1 = max_pool_2X2(h_conv1)
    # lrn处理, 模拟生物的侧抑制机制，使得响应大的值更大，并且抑制其他反馈较小的神经元。
    # 增强模型的泛化能力（只对relu这种没有上界边界的激活函数有用，对softmax无用）
    norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    # 开始第二次卷积

    # 最后一个64表示卷积核的数量
    w_conv2 = weight_variable_l2([5, 5, 32, 64], stddev=5e-2, wl=0.0)
    b_conv2 = bias_variable([64])
    # 第二卷积  14*14
    h_conv2 = tf.nn.relu(conv2d(norm1, w_conv2) + b_conv2)
    # lrn处理
    norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    # 第二池化 7*7
    h_pool2 = max_pool_2X2(norm2)


    # 对第二卷积层的输出进行变形
    h_pool2_flat = tf.reshape(h_pool2, [batch_size, 7*7*64])
    # 获得位数
    dim = h_pool2_flat.get_shape()[1].value
    # 全连接层
    w_fc1 = weight_variable_l2([dim, 1024], stddev=0.04, wl=0.004)
    b_fc1 = bias_variable([1024])
    # 加一个隐层
    hidden1 =  tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    # dropout
    keep_prob = tf.placeholder(tf.float32)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob=keep_prob)

    # 加一个隐层连接层
    w_fc2 = weight_variable_l2([1024, 512], stddev=0.04, wl=0.004)
    b_fc2 = bias_variable([512])
    hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, w_fc2)+b_fc2)
    # 输出层
    w_fc3 = weight_variable_l2([512, 10], stddev=1/512, wl=0.0)
    b_fc3 = bias_variable([10], value=0.0)
    y = tf.nn.softmax(tf.matmul(hidden2, w_fc3) + b_fc3)
    # logits = tf.add(tf.matmul(hidden2, w_fc3), b_fc3)
    #
    # loss = loss(logits, y_true)
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
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        if i%100==0:
            train_accu = accuracy.eval({x: batch_xs, y_true: batch_ys, keep_prob: 1})
            print('step %d, train_accuracy %g'%(i, train_accu))
        opt.run({
            x: batch_xs, y_true: batch_ys, keep_prob: 0.5
        })
    # PREDICTING

    print(accuracy.eval({x: mnist.test.images, y_true: mnist.test.labels, keep_prob: 1}))


