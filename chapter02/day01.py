#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day01.py
@time: 2019/03/06
"""
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
'''
numpy 数组的基本操作
'''
def basic_opration1():
    # 按照一定步长创建数组
    ara = np.arange(0, 10, 2)
    print(type(ara))
    print(ara)
    # 创建多维度的数组, 里面的数据类型一定要一致
    arr = np.array((ara, ara, ara))
    print(type(arr))
    print(arr.shape)
    # 通过 arr[m,n]模式选取对应的多维数组的元素(对应的行列)
    print(arr[0, 0])

    # 浮点数可以转换为复数,.
    f = 1.23
    cf = np.complex(f)
    print(cf)
    print(float(True))
    print(int(True))
    # 自定义数据类型,一定需要
    my_type = np.dtype([('name', str, 30), ('sale_price', np.float), ('real_price', np.float)])

    goods = np.array([('books', 35.67, 23.45), ('bikecycle', 234.5, 220)], dtype=my_type)
    print(goods[1]['name'])
    print(goods[1])

    # 数组的切片的操作
    a = np.arange(0, 10, 1)
    print(a[1:6:2])

    b = np.arange(100).reshape(2, 5, 10)
    # b[第一个维度,第二个维度,第三个维度]
    print(b[:, ::-1, :])

    c = np.arange(20).reshape(-1, 1)
    print(c.shape)
    # 将n维的数组拉成一维的数组
    d = c.ravel()
    print(d.shape)
    print(c.shape)
    e = c.flatten()
    print(e.shape)
    print(c.shape)
    # transpose(), 转置操作
    print(c.transpose())
    # np.fill(number)填充数组
    averages = np.zeros(10)
    print(averages)
    averages.fill(2)
    print(averages)
'''
数据的组合
'''
def basic_stack():
    a = np.arange(9).reshape(3, 3)
    b = a * 2
    print(b)
    # 水平组合
    c = np.hstack((a, b))
    print(c)
    d = np.concatenate((a, b), axis=1)
    print(d)
    # 垂直合拼
    e = np.vstack((a, b))
    print(e)
    # 深度合并, 深度合并需要两个数组的元素维度相同， 每个位置上的元素对应合并
    print(np.dstack((a, b)))
    # 也是，列合并
    print(np.column_stack((a, b)))
    # 行=组合
    print(np.row_stack((a, b)))
'''
数据的分割
'''
def basic_split():
    a = np.arange(8)

    b = a.reshape(2, 4)
    print(b)
    # 水平方向的分割，一定需要列数是偶数
    b1 = np.hsplit(b, 2)
    print(b1)
    print(b1[1].shape)
    # 垂直分割
    b2 = np.vsplit(b, 2)
    print(b2)
    print(b2[1].shape)
    # 深度分割
    ar = np.arange(20).reshape(5, 2, 2)
    print(np.dsplit(ar, 2)[1].shape)

if __name__ == '__main__':
    a = np.array([1,2,3,4,5])
    b = a.astype(np.float)
    print(a.ndim)
    print(a.size)
    print(b.tolist())
    print(a.tolist())







