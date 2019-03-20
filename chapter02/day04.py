#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day04.py
@time: 2019/03/12
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
'''
线性模型
'''
'''
日期转数字
'''
def datestr2num(s):
    # 字节转str
    s = s.decode()
    return datetime.datetime.strptime(s, '%Y-%m-%d').date().weekday()


def basic_use():
    c, dates, volume = np.loadtxt('000032.csv', skiprows=1, delimiter=',', converters={0: datestr2num},
                                  usecols=(4, 0, 5), unpack=True)
    c = c[::-1][-10:]

    print(c)
    print(c.compress(c > 6.7))
    print(c[c > 6.7])
    print(np.where(c > 6.7, 0, 1))
    # 计算阶乘
    a = np.arange(1, 10, 1)
    # a = np.linspace()
    print(a)
    prod = a.prod()
    print(prod)
    # 们想知道1~8的所有阶乘值呢
    print(a.cumprod())

if __name__ == '__main__':
    c, dates, volume = np.loadtxt('000032.csv', skiprows=1, delimiter=',', converters={0: datestr2num},
                                  usecols=(4, 0, 5), unpack=True)

    c1, dates1, volume1 = np.loadtxt('000561.csv', skiprows=1, delimiter=',', converters={0: datestr2num},
                                  usecols=(4, 0, 5), unpack=True)

    # 计算相关系数
    corr = np.corrcoef(c[-400:], c1[-400:])
    print(corr)
    '''
    我们使用 polyfit 函数对数据进行了多项式拟合。我们学习使用 polyval 函数计算多项式的
取值,使用 roots 函数求得多项式函数的根,以及 polyder 函数求解多项式函数的导函数。
    '''
    # sign函数返回数组中每个元素的正负符号。
    b = np.arange(-3, 6)
    sign = np.sign(b)
    print(sign)



