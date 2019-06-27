#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: lstm_tensorflow.py
@time: 2019/05/04
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import tensorflow as tf
import reader



class PTBInput(object):
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        # ｌｓｔｍ展开的步长
        self.num_steps = num_steps = config.num_steps
        # 每轮迭代的需要训练的次数
        self.epoch_size = ((len(data)//batch_size)-1)//num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name
        )


class PYBModel(object):

    def __init__(self, is_training, config, input_=PTBInput()):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        # ｌｓｔｍ的节点数
        size = config.hidden_size
        # 词汇表的大小
        vocab_size = config.vocab_size

        def lstm_cell():
            '''
            forget_bias 表示是forget gate的bias 是０
            state_is_tuple　＝　ＴＲＵＥ表示接收和返回的是　2　tuple 的类型
            :return:
            '''
            return tf.contrib.rnn.BasicLSTMCell(
                size, forget_bias=0.0, state_is_tuple=True
            )
        cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob
                )
            cell = tf.contrib.rnn.MultiRNNCell(
                [attn_cell() for _ in range(config.num_layers)],
                state_is_tuple = True
            )
        # lstm单元可以读入一个单词，并且结合之前存储的状态state计算下一个单词的出现的概率分布
        #并且每读入一个单词，它的状态就会更新
        self.initial_states = cell.zero_state(batch_size, tf.float32)
        with tf.device('/cpu:0'):
            # 初始化 embedding矩阵，　行数　列数设置为hidden_size和ｌｓｔｍ中隐含的节点一致
            embedding = tf.get_variable(
                'embedding', [vocab_size, size], dtype=tf.float32
            )
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

