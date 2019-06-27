#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: auto_decode_M.py
@time: 2019/05/02
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf



def xavier_init(fan_in, fan_out, constant=1):
    '''

    :param fan_in:
    :param fan_out:
    :param constant:
    :return:权重满足高斯分布的变量
    '''
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    # The generated values follow a uniform distribution in the range
    #   `[minval, maxval)]
    return  tf.random_uniform((fan_in, fan_out), minval=low,
                              maxval=high, dtype=tf.float32)


class GaussianNoiseAutoencoder(object):


    def __init__(self, n_input, n_hiddle, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        '''

        :param n_input: 输入的变量数
        :param n_hiddle: 隐含层的节点数
        :param transfer_function: 隐含层的激活函数
        :param optimizer:
        :param scale: 高斯噪声系数
        '''
        self.n_input = n_input
        self.n_hiddle = n_hiddle
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 加入高斯噪声计算隐层的预测值
        self.hiddle = self.transfer(tf.add(tf.matmul((self.x+self.scale*tf.random_normal((n_input,)))
                                                     , self.weights['w1']), self.weights['b1']))
        # 复原数据
        self.reconstruction = tf.add(tf.matmul(self.hiddle, self.weights['w2']),
                                               self.weights['b2'])
        # loss
        self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x),
                                             2.0))
        # optimizor
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hiddle))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hiddle], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hiddle, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, x):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={
            self.x:x, self.scale: self.training_scale
        })
        return cost

    def cacl_total_cost(self, x):
        cost = self.sess.run(self.cost, feed_dict={
            self.x:x, self.scale: self.training_scale
        })
        return cost

    # 返回自编码器隐层的输出结果, 获得原始信息的高阶特性
    def transform(self, x):
        return self.sess.run(self.hiddle, feed_dict={self.x:x,
                                                     self.scale:self.training_scale})
    #将隐层的输出的结果作为输入，将隐层提取的高阶的信息浮复原为原始数据
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={
            self.hiddle: hidden
        })
    # 运行复原过程
    def reconstruct(self, x):
        return self.sess.run(self.reconstruction, feed_dict={
            self.x:x, self.scale:self.training_scale
        })

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index+batch_size)]


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

   # batch_xs, batch_ys = mnist.train.next_batch(100)

    ss_x = StandardScaler()

    ss = ss_x.fit(mnist.train.images)
    x_train = ss.transform(mnist.train.images)
    x_test = ss.transform(mnist.test.images)

    #x_train, x_yan = train_test_split(x_train, test_size=0, random_state=42)
    n_samples = int(mnist.train.num_examples)
    train_epochs = 20
    batch_size = 128
    display_step = 1

    autoencoder = GaussianNoiseAutoencoder(n_input=784,
                                n_hiddle=200,transfer_function=tf.nn.softplus,
                                optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                           scale=0.01)

    for epoch in range(train_epochs):
        avg_cost = 0
        total_batch = int(n_samples/batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(x_train, batch_size)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples*batch_size
        if epoch % display_step ==0 :
            print('epoch:', '%04d' %(epoch+1), 'cost =',
                  '{:.9f}'.format(avg_cost))

    print('total cost:' + str(autoencoder.cacl_total_cost(x_test)))