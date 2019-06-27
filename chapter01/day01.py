#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day01.py
@time: 2019/03/02
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import calendar


def buildConnectionString(params):
    # 不能使用join连接非字符串
    return ';'.join('%s=%s' % (k,v) for k, v in params.items())


'''
字典的一些常用的操作
'''
def for_dict():
    params = {
        'server': 'mpilgrim',
        'database': 'master',
        'pwd': 'secret',
        'server1': 'mpilgrim',
        'database1': 'master',
        'pwd1': 'secret'

    }

    for (k, v) in params.items():
        print(k, '->>', v)
    print('修改dict')
    params['pwd1'] = '652'
    params['Pwd1'] = '652'
    print('删除特定的元素')
    del params['pwd1']

    for (k, v) in params.items():
        print(k, '->>', v)
    print('清除所有的元素')
    params.clear()
    print(params)


'''
列表的一些常用的操作
'''
def for_list():
    _list = ['1', 'fa', 'yuan', ['tu', 'chen'], {'one': 'mars', 'two': 'lily'}]
    # 反转列表
    _list.reverse()
    print(_list)
    # 插入值
    _list.append('test')
    print(_list)
    # extend方法参数只能是list类型
    _list.extend(['test', 'run'])
    print(_list)

    # 测试数据是否存在
    print('fa' in _list)
    print('fato' not in _list)
    print('****')
    for one in _list:
        print(one)
        print('tu' in one)

    # 删除一个值
    _list.remove('fa')
    # pop 出栈最后的一个元素
    _list.pop()
    print(_list)

if __name__ == '__main__':



    print(calendar.THURSDAY)
    result = [one**2 for one in [1,2,3,4,5]]
    print(result)

    #lambda one : if one >10  return one

    other = [one for one in result if one < 10]
    other1 = [print(item) for item in result if item >10 ]
    print(other)
    print(':'.join(['34','35', '43']))
