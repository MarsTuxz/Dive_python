#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day04.py
@time: 2019/03/18
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import scipy as sp
import datetime


'''
组合投资，组合dataFrame(merge的使用)
 找到NAN 数据所在的行和列

'''
def Portfolio():
    '''
        组合投资，组合dataFrame
        '''
    data1 = pd.read_csv('000032.csv', usecols=(0, 2), index_col='date')
    data1 = data1[::-1]
    data1.columns = ['32']
    data2 = pd.read_csv('000561.csv', usecols=(0, 2), index_col='date')
    data2 = data2[::-1]
    data2.columns = ['561']
    # merge 要求两个df的index一样
    final_data = pd.merge(data1, data2, how='inner', left_index=True, right_index=True)
    # print(final_data)
    print(final_data[:10])
    '''
    找到NAN 数据所在的行和列
    '''
    nan = np.where(np.isnan(final_data))
    print(nan)
    print(nan[0])
    print(nan[1])


def t_f_check():
    '''
    T-检验，F-检验
    '''
    # 测试股票的日收益回报是否等于0
    close = pd.read_csv('000032.csv', usecols=(3,))
    close = close[::-1]
    close = np.copy(close)
    ret = close[1:]/close[:-1]-1
    ret = ret.ravel()
    print('mean and T-value, p-value')
    # p值表示0假设出现的概率
    print(round(sp.mean(ret), 4), stats.ttest_1samp(ret, -0.001))
    # 检验方差是否相等 bartlett() F-检验
    close1 = pd.read_csv('000561.csv', usecols=(3,))
    close1 = close1[::-1]
    close1 = np.copy(close1)
    ret1 = close1[1:] / close1[:-1] - 1
    ret1 = ret1.ravel()
    print(stats.bartlett(ret, ret1))

if __name__ == '__main__':
    '''
    一月效应
    '''
    data = pd.read_csv('000032.csv', usecols=(3, 0), index_col=0)
    close = data['close'][::-1]
    close = np.copy(close)
    ret = close[1:] / close[:-1] - 1
    date = data.index.to_list()
    month_date = []
    for i in range(len(date)-1):
        month = date[i].split('/')
        #print(month)
        t = ''.join((month[0], month[1]))
        month_date.append(datetime.datetime.strptime(t, '%Y%m'))

    y = pd.DataFrame(ret, index=month_date, columns=['month_ret'])
    ret_monthly = y.groupby(y.index).sum()
    #拿到所有的1月份的数据
    #print(np.where(ret_monthly.index.month == 2)[0])
    #print(ret_monthly.index.month == 2)
    january_ret = ret_monthly[ret_monthly.index.month == 2]
    print(january_ret)




    other_ret = ret_monthly[ret_monthly.index.month != 2]
    print('january_ret: ', np.mean(january_ret))
    print('other_ret: ', np.mean(other_ret))
    print(january_ret.values)
    print(stats.bartlett(january_ret.values.ravel(), other_ret.values.ravel()))



