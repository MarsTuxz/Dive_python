#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day03.py
@time: 2019/03/15
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def basic_pd():
    # 生成日期
    dates = pd.date_range('20130101', periods=5)
    np.random.seed(200)
    x = pd.DataFrame(np.random.rand(5, 2), index=dates, columns=('a', 'b'))
    print(x)
    print(x.describe())

    print(x.index.to_list())


def cul_return():
    '''
        计算月对数收益率， 百分收益率之间的关系：return = exp(logret)-1
        '''
    data = pd.read_csv('000032.csv', index_col='date')
    data = data[::-1]
    close = np.copy(data.get('close'))
    # 计算对数回报率
    logret = np.log(close[1:] / close[:-1])
    yyyymm = []
    d0 = data.index.to_list()
    # print(d0)
    for i in range(0, np.size(logret)):
        ymd = d0[i].split('-')
        yyyymm.append(''.join([ymd[0], ymd[1]]))
        # 如果日期是date形式
        # yyyymm.append(''.join([d0[i].strftime('%Y'), d0[i].strftime('%m')]))
    y = pd.DataFrame(logret, yyyymm, columns=['ret_monthly'])
    # 根据条件组合排序
    ret_monthly = y.groupby(y.index).sum()
    print(ret_monthly)

    '''
    计算年收益率
    '''
    data = pd.read_csv('000032.csv', index_col='date')
    data = data[::-1]
    close = np.copy(data.get('close'))
    # 计算对数回报率
    logret = np.log(close[1:] / close[:-1])
    yyyymm = []
    d0 = data.index.to_list()
    # print(d0)
    for i in range(0, np.size(logret)):
        ymd = d0[i].split('-')
        yyyymm.append(ymd[0])
        # 如果日期是date形式
        # yyyymm.append(''.join([d0[i].strftime('%Y'), d0[i].strftime('%m')]))
    y = pd.DataFrame(logret, yyyymm, columns=['ret_yearly'])
    print(y[:5])
    ret_monthly = y.groupby(y.index).sum()
    print(ret_monthly)

if __name__ == '__main__':
    cul_return()


