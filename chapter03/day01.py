#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day01.py
@time: 2019/03/14
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import scipy.optimize as optimize



def my_f(x):
    return 4+x**2

'''
线性 回 归 和 资本资产定价模型 ( CAPM )
根据著名 的 CAPM 模 型 , 单只 股 票 的 回报率和 市场 的 回报 率线性 相 关 。 通 常情况 下 ,
我们 考虑 股 票 的 超额 回 报率与 市 场 的 超 额 回 报 率之 间 的 关 系 。
'''
def CAMP():
    stock_ret = [0.05, 0.01, -0.07, -0.02]
    mkt_ret = [0.005, 0.009, -0.001, 0.023]
    beta, alpha, r_value, p_value, std_err = stats.linregress(stock_ret, mkt_ret)


if __name__ == '__main__':
    # 优化算法， 给定目标函数和约束条件选择最佳的投资组合
    #print(optimize.fmin(my_f, 5))
    data = pd.read_csv('000032.csv')
    # 统计各个类别分类的数目
    print(data['close'].value_counts())
    #　各个类别的统计信息
    print(data.info())
    print(data.describe())
    #

