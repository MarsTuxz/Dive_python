#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day05.py
@time: 2019/03/19
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime
import scipy as sc

'''
52周(一年的时间)换手策略
'''
def _52_week_change():
    data = pd.read_csv('000032.csv', index_col=0)
    data = data[::-1]
    #print(data.index[-1:][0])
    enddate = data.index[-1:][0]
    temp_data = enddate
    temp_data = datetime.strptime(temp_data, '%Y/%m/%d')
    print(temp_data)
    #enddate = datetime.now()
    begdate = temp_data - relativedelta(years=1)
    begdate = begdate.strftime('%Y/%m/%d')
    print(begdate)
    year_data = data[begdate: enddate]
    #print(year_data[-10:])
    high = np.max(year_data['high'])
    low = np.min(year_data['low'])
    print('today date:', enddate)
    print(low, high)

'''
用 Roll ( 1984 ) 模型来估算买卖价差(主要是用来反应股票的流动性指标的)
S = 2 sqrt( cov (~pi,~p(i-1) ))
'''
def roll():
    data = pd.read_csv('000032.csv', index_col=0)
    data = data[::-1]
    # print(data.index[-1:][0])
    enddate = data.index[-1:][0]
    temp_data = enddate
    temp_data = datetime.strptime(temp_data, '%Y/%m/%d')
    print(temp_data)
    # enddate = datetime.now()
    begdate = temp_data - relativedelta(months=2)
    begdate = begdate.strftime('%Y/%m/%d')
    print(begdate)
    print(enddate)
    print(data.index)

    '''
    使用这种办法的时候一定要注意，如果索引是string的类型，那么如果是日期字符串的类型的，
    那么比较的时候： 2018/09/13 < 2018/1/12
    '''
    month_data = data[data.index>=begdate]
    month_data = month_data[month_data.index<=enddate]
    print(month_data)
    month_data_close = month_data['close'].values

    d = np.diff(month_data_close)
    cov_ = sc.cov(d[:-1], d[1:])
    print(cov_)
    if cov_[0, 1]<0:
        print('roll spread for negetive', round(2*sc.sqrt(-cov_[0,1]), 3))
    else:
        print('roll spread for positive', round(cov_[0, 1]))

def roll_true():
    data = pd.read_csv('000032.csv', index_col=0, parse_dates=True)
    data = data[::-1]
    # print(data.index[-1:][0])
    enddate = data.index[-1:][0]
    begdate = enddate - relativedelta(months=2)
    print(begdate)
    print(enddate)
    month_data = data[data.index >= begdate]
    month_data = month_data[month_data.index <= enddate]
    print(month_data)
    month_data_close = month_data['close'].values

    d = np.diff(month_data_close)
    print(d)
    cov_ = sc.cov(d[:-1], d[1:])
    print(cov_)
    if cov_[0, 1] < 0:
        print('roll spread for negetive', round(2 * sc.sqrt(-cov_[0, 1]), 3))
    else:
        print('roll spread for positive', round(cov_[0, 1]))


'''
用 Amihud ( 2002 )模型来估算反流动性指标
回归模型
'''
def Amihud():
    data = pd.read_csv('000032.csv', index_col=0, parse_dates=True)
    data = data[::-1]
    # print(data.index[-1:][0])
    enddate = data.index[-1:][0]
    begdate = enddate - relativedelta(months=2)
    print(begdate)
    print(enddate)
    month_data = data[data.index >= begdate]
    month_data = month_data[month_data.index <= enddate]
    # dataframe 。value的格式是numpy格式。
    close = month_data.close.values
    print(type(close))
    print(close.shape)
    dollar_vol = np.array(month_data.volume*close)
    print(dollar_vol.shape)
    ret = np.array(close[1:]/close[:-1]-1)
    print(ret.shape)
    illiq = np.mean(np.divide(abs(ret), dollar_vol[1:]))
    print('Aminud illiq=', illiq)


'''
Pastor 和 Stambaugh ( 2003 ) 流动性指标
'''
def Pastor_Stambaugh():
    data = pd.read_csv('000032.csv', index_col=0, parse_dates=True)
    data = data[::-1]
    data1 = pd.read_csv('000032.csv', index_col=0, parse_dates=True)
    data1 = data1[::-1]
    # print(data.index[-1:][0])
    enddate = data.index[-1:][0]
    begdate = enddate - relativedelta(months=1)
    print(begdate)
    print(enddate)
    month_data = data[data.index >= begdate]
    month_data = month_data[month_data.index <= enddate]

    month_data1 = data1[data1.index >= begdate]
    month_data1 = month_data1[month_data1.index <= enddate]
    # 市场回报率
    ret = month_data.close[1:].values/month_data.close[:-1].values - 1

    dollar_vol = np.array(month_data.close[1:])*np.array(month_data.volume[1:])
    date = []
    d0 = month_data.index
    for i in range(0, np.size(ret)):
        date.append(''.join([d0[i].strftime('%Y'), d0[i].strftime('%m'), d0[i].strftime('%d')]))

    tt = pd.DataFrame(ret, np.array(date, dtype=np.int64), columns=['ret'])
    tt2 = pd.DataFrame(dollar_vol, np.array(date, dtype=np.int64), columns=['dollar_vol'])
    tt3 = pd.merge(tt, tt2, left_index=True, right_index=True)
    # 无风险利率
    safe_retio = np.arange(0, len(tt), dtype=float)
    safe_retio.fill(0.0002)
    safe_tt = pd.DataFrame(safe_retio, np.array(date), columns=['Rf'])
    final = pd.merge(tt3, safe_tt, left_index=True, right_index=True)
    # 市场收益率
    x1 = month_data1.close.values[:-1]
    # 带符号的交易额
    x2 = np.sign(np.array(final.ret[:-1]-final.Rf[:-1]))*np.array(final.dollar_vol[:-1])
    y = final.ret[1:] - final.Rf[1:]
if __name__ == '__main__':
    Pastor_Stambaugh()