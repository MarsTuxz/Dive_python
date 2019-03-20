#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day02.py
@time: 2019/03/08
"""
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import datetime
'''
分析日期数据
'''


def common_fun():
    # 单位矩阵
    i2 = np.eye(4)
    print(i2)
    # 数据存储文件
    np.savetxt('eye.txt', i2)
    # 读取文件
    c, v = np.loadtxt('000032.csv', skiprows=1, delimiter=',', usecols=(4, 5), unpack=True)
    # print(c)
    # 计算成交量加权平均价格（VWAP）
    vwap = np.average(c, weights=v)
    print('vwap = ', vwap)
    # j计算算术平均值
    wap = np.mean(c)
    print('算术平均值为：', wap)
    # 计算时间加权平均价格
    print(c.size)
    t = np.arange(c.size)
    twap = np.average(c, weights=t)
    print('twap = ', twap)
    # 计算区间的中间点
    print('middle is ', np.max(c) - np.min(c))
    print('middle is ', np.ptp(c))
    # 获得中位数
    median = np.median(c)
    print('中位数为：', median)
    # 获得数组的排序
    sorted_c = np.msort(c)
    print('sorted_c:', sorted_c[c.size // 2])
    print('sorted_c:', sorted_c[(c.size - 1) // 2])
    # 计算方差
    variance = np.var(c)
    print('variance is ', variance)

'''
计算收益率
'''
def return_ratio():
    c, v = np.loadtxt('000032.csv', skiprows=1, delimiter=',', usecols=(4, 5), unpack=True)
    diff_c = np.diff(c)
    # 收益率计算
    return_ratio = diff_c / c[:-1]
    # 计算收益率的平局值
    mean_r_r = np.mean(return_ratio)
    print(mean_r_r)
    # 计算收益率的中位数
    median_r_r = np.median(return_ratio)
    print(median_r_r)
    # 计算收益率的方差
    std_r_r = np.std(return_ratio)
    print(std_r_r)
    print(np.max(return_ratio))

    # 计算对数收益率
    logreturns = np.diff(np.log(c))
    # 计算波动率
    annual_volatility = np.std(logreturns) / np.mean(logreturns)
    # 计算日波动率
    annual_volatility = annual_volatility / np.sqrt(1. / 252.)
    print('annual_volatility: ', annual_volatility)


'''
日期转数字
'''
def datestr2num(s):
    # 字节转str
    s = s.decode()
    return datetime.datetime.strptime(s, '%Y-%m-%d').date().weekday()


if __name__ =='__main__':

    # 日期分析
    c, dates, volume = np.loadtxt('000032.csv', skiprows=1, delimiter=',', converters={0: datestr2num}, usecols=(4, 0, 5), unpack=True)
    averages = np.zeros(5)
    # 保存各个工作日的收盘价
    for i in range(5):
        indices = np.where(dates==i)
        print(indices)
        close_prices = np.take(c, indices)
        part_volume = np.take(volume, indices)
        #averages[i] = close_prices
        avg = np.mean(close_prices)
        print("Day", i,  "Average ", avg)

        vwap = np.average(close_prices, weights=part_volume)
        print("Day", i, "vwap ", vwap)

        t = np.arange(close_prices.size)
        #计算数组的维度需要和权重的维度一致
        twap = np.average(close_prices.ravel(), weights=t)
        print("Day", i, "twap ", twap)








