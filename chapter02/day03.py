#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day03.py
@time: 2019/03/12
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

'''
分析周数据
'''

'''
日期转数字
'''
def datestr2num(s):
    # 字节转str
    s = s.decode()
    return datetime.datetime.strptime(s, '%Y-%m-%d').date().weekday()

'''
计算每个周的总的数据
'''
def summarize(a, o, h, l, c):
    monday_open = o[a[0]]
    week_high = np.max( np.take(h, a) )
    week_low = np.min( np.take(l, a) )
    friday_close = c[a[-1]]
    return (monday_open, week_high, week_low, friday_close)


'''

计算真实波动幅度均值（ATR）'''

def get():
    # 计算真实波动幅度均值（ATR）
    low, dates, volume, high, close = np.loadtxt('000032.csv', skiprows=1, delimiter=',', converters={0: datestr2num},
                                                 usecols=(4, 0, 5, 2, 3), unpack=True)
    low = low[-20:][::-1]
    dates = dates[-20:][::-1]
    print(dates)
    high = high[-20:][::-1]
    close = close[::-1]
    # 获得前一个脚力日的收盘价
    previousclose = close[-21:-1]
    '''
    当日股价范围,即当日最高价和最低价之差。
 h – previousclose 当日最高价和前一个交易日收盘价之差。
 previousclose – l 前一个交易日收盘价和当日最低价之差。
    '''
    # maximum 比较同一个行的多个维度的数据的最大值
    truerange = np.maximum(high - low, high - previousclose, previousclose - low)
    print(truerange)



'''
获得周汇总数据
'''

def get_weekData():
    # 日期分析
    c, dates, volume = np.loadtxt('000032.csv', skiprows=1, delimiter=',', converters={0: datestr2num},
                                  usecols=(4, 0, 5), unpack=True)
    c = c[::-1]
    dates = dates[::-1]
    # close = c[:16]
    # dates = dates[:16]
    # 找到第一个星期一
    first_monday = np.ravel(np.where(dates == 0))[0]
    print(dates[:3])
    print("The first Monday index is", first_monday)
    # 找到最后一个周五
    last_friday = np.ravel(np.where(dates == 4))[-1]
    print(dates[:3])
    print("The last friday index is", last_friday)

    # weeks_indices = np.arange(first_monday, last_friday + 1)
    dates = dates[first_monday: last_friday + 1]
    c = c[first_monday: last_friday + 1]

    print("Weeks indices initial", dates)

    # 获得每周五的index

    friday_list = np.ravel(np.where(dates == 4))
    print(friday_list[:2])
    # apply_along_axis 函数。
    # 这个函数会调用另外一个由我们给出的函数,作用于每一个数组元素上
    # 通过split函数划分周末
    split_friday_list = list(map(lambda x: x + 1, friday_list))
    print('*******')
    print(split_friday_list)
    # 前提是周五一定存在
    # split 函数可以通过截取对应的区间上的数据

    weeks_close_index = np.split(c, split_friday_list)
    print(weeks_close_index)
    weeksummary = np.apply_along_axis(summarize, 1, weeks_close_index)



if __name__ =='__main__':
    pass





