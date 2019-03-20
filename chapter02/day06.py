#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day06.py
@time: 2019/03/14
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
'''
排序函数
'''



def datestr2num(s):
    # 字节转str
    s = s.decode()
    return datetime.datetime.strptime(s, '%Y-%m-%d').date().weekday()

'''
numpy 专用函数
'''
def special_fun():
    c, dates, volume = np.loadtxt('000032.csv', skiprows=1, delimiter=',', converters={0: datestr2num},
                                  usecols=(4, 0, 5), unpack=True)

    print(c[:20])
    data = (dates, c)
    # 使用lexsort函数排序。数据本身已经按照日期排序, 不过我们现在优先按照收盘价排序:
    indices = np.lexsort((dates, c))
    # argmax 函数返回数组中最大值对应的下标
    print(np.argmax(c))
    # nanargmax 函数提供相同的功能,但忽略NaN值
    print(np.nanargmax(c))
    # argwhere 函数根据条件搜索元素,并分组返回对应的下标
    print(np.argwhere(c > 18))

    # searchsorted 函数为指定的插入值返回一个在有序数组中的索引位置,从这个位置插入可
    # 以保持数组的有序性
    a = np.arange(5)
    index = np.searchsorted(a, [-1, 6])
    new_a = np.insert(a, index, [-1, 6])
    print(new_a)

    # NumPy的 extract 函数可以根据某个条件从数组中抽取元素
    # $ 使用 extract 函数基于生成的条件从数组中抽取元素:
    a = np.array([1, 2, -4, 43, 4, -2, 5, 5, 6, 0, 0, 1])
    condition = a > 0
    print(np.extract(condition, a))
    # 使用 nonzero 函数抽取数组中的非零元素
    print(np.nonzero(a))
    print(a[np.nonzero(a)])

'''
numpy 金融函数
'''
def financial():
    '''
        假设你贷款100万,年利率为10%,要用30年时间还完贷款,那么每月你必须支付多少资金
        呢?我们来计算一下。
        使用刚才提到的参数值,调用 pmt 函数。
        print "Payment", np.pmt(0.10/12, 12 * 30, 1000000)
        计算出的月供如下所示:
        Payment -8775.71570089
        '''
    print("Payment", np.pmt(0.10 / 12, 12 * 30, 3000000))
    '''
    考虑贷款9 000,年利率10%,每月固定还款为100的情形。
    通过 nper 函数计算出付款期数。
    print "Number of payments", np.nper(0.10/12, -100, 9000)
    计算出的付款期数如下:
    '''
    print("Number of payments", np.nper(0.10 / 12, -100, 9000))
    print("Number of payments", np.nper(0.10 / 12, -10000, 1000000))


if __name__ == "__main__":
    '''
    各种窗函数
    '''
    # 巴特利特窗(Bartlett window)是一种三角形平滑窗
    window = np.bartlett(42)

    # 布莱克曼窗(Blackman window)形式上为三项余弦值的加和
    window = np.blackman(48)
    # 汉明窗(Hamming window)形式上是一个加权的余弦函数
    window = np.hamming(148)
    # 凯泽窗(Kaiser window)是以贝塞尔函数(Bessel function)定义的
    window = np.kaiser(42, 14)

    plt.plot(window)

    #绘制 sinc 函数
    np.outer()

    plt.show()



