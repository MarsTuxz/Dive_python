#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day02.py
@time: 2019/03/14
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from matplotlib.dates import DateFormatter, WeekdayLocator, HourLocator, MONDAY, DayLocator
from mpl_finance import plot_day_summary_oclh, _candlestick

'''
金融方面的一些图形的做法
'''

def datestr2num(s):
    # 字节转str
    s = s.decode()
    return datetime.datetime.strptime(s, '%Y-%m-%d').date().weekday()

'''
杜邦等式把股本回报率 ( ROE )
分 为 3 个 比 率 : 毛 利 率 、 资 产 周 转 率 和 权益 乘 数 。
'''
def dupont_equation():

    ticker = 'Ticker'
    name1 = 'profitMargin'
    name2 = 'assertTurnover'
    name3 = 'equitMultiplier'
    scale = 7 # scale the 1 s t ratio
    raw_data = {ticker: ['sltx', 'kdxf', 'albb'],
                name1:[0.1467*scale, 0.0671*scale, 0.2*scale],
                name2:[0.899, 1.190, 2.45],
                name3:[6.32, 4.55, 2.566]}
    df = pd.DataFrame(raw_data)
    f, ax = plt.subplots(1, figsize=(10, 5))
    w = 0.745
    x = [i+1 for i in range(len(df[name1]))]
    print(x)

    tick_pos = [i + (w / 2.) for i in x ]
    ax.bar(x, df[name1], width=w, label=name1, alpha=0.5,
           color='blue')
    ax.bar(x, df[name2], width=w, bottom=df[name1], label=name2, alpha=0.5,
           color='red')
    ax.bar(x, df[name3], width=w, bottom=[i+j for i,j in zip(df[name1]
                                                             ,df[name2])], label=name3, alpha=0.5,
           color='green')
    plt.xticks(tick_pos, df[ticker])
    plt.ylabel('Dupoint Identity')
    plt.xlabel('Different tickers')
    plt.legend(loc='upper right')
    plt.title('" DuPont Identity for 3 fi rms "')
    plt.show()


'''
组合投资波动率比较
'''
def portfolio_comparison():
    year = [2013,2014,2015,2016,2017]
    ret_A = np.array([0.12, 0.3,0.01, -0.2, 0.02])
    ret_B = np.array([-0.12, 0.13,0.11, -0.12, 0.12])
    port_EW = (ret_A+ ret_B) / 2
    plt.plot(year, ret_A, lw=2, c='blue', label='jd')
    plt.plot(year, ret_B, lw=2, c='yellow', label='tb')
    plt.plot(year, port_EW, lw=2, c='red', label='portfolio')
    plt.xlabel('year')
    plt.ylabel('Returns')
    plt.show()


'''
用直方图 显示收益 率分布
用历史 回报率的均值来估 计期望收
'''
def return_ratio_distribution(close):
    close_ratio = np.diff(close)/close[:-1]
    plt.hist(close_ratio, 100)
    plt.show()

'''
蜡烛图画法
'''
def candle_figure():
    data = pd.read_csv('000032.csv')
    open = data['open']
    high = data['high']
    close = data['close']
    low = data['low']
    dates = np.arange(len(data))
    print(dates[:10])
    volume = data['volume']
    ticker = 'ALBB'
    mondays = WeekdayLocator(MONDAY)
    alldays = DayLocator()
    weekFormatter = DateFormatter('%b %d')
    dayFormatter = DateFormatter('%d')
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)
    ax.xaxis.set_minor_formatter(dayFormatter)
    plot_day_summary_oclh(ax, (dates, open, close, high, low))
    _candlestick(ax, (dates, open, close, high, low), width=0.6)
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=80, horizontalalignment='right')
    plt.show()


def datestr2num(s):
    # 字节转str
    s = s.decode()
    return datetime.datetime.strptime(s, '%Y-%m-%d').date().weekday()



'''
比较一个会计年个股对数收益率表现
'''
def performance_stock():
    c, dates, volume = np.loadtxt('000032.csv', skiprows=1, delimiter=',', converters={0: datestr2num},
                                  usecols=(4, 0, 5), unpack=True)
    c1, dates1, volume1 = np.loadtxt('000561.csv', skiprows=1, delimiter=',', converters={0: datestr2num},
                                  usecols=(4, 0, 5), unpack=True)
    c = c[::-1][-220:]
    logret = np.log(c[1:]/c[:-1])
    sum_logret = np.exp(sum(logret))-1

    c1 = c1[::-1][-220:]
    logret1 = np.log(c1[1:] / c1[:-1])
    sum_logret1 = np.exp(sum(logret1)) - 1
    y_pos = np.arange(2)
    plt.barh(y_pos, (sum_logret, sum_logret1), left=0, alpha=0.3)
    plt.show()


if __name__ == '__main__':
    c, dates, volume = np.loadtxt('000032.csv', skiprows=1, delimiter=',', converters={0: datestr2num},
                                  usecols=(4, 0, 5), unpack=True)
    # c = np.diff(c)
    # volume = np.diff(volume)
    #volume = volume[1:]
    #plt.scatter(volume, c)
    #plt.show()


    '''
    了 解简单利 率 和 复利 利 率
    简单利 率 不考虑利息的利息, 而 复利利率考虑
    FV( 简单利率) =本金×（1+ratio×n）
    FV( 复利利率) =The principal * (1+ratio)^n
    '''
    performance_stock()
    a = b = 19
