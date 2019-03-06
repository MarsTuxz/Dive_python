#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day04.py
@time: 2019/03/04
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd


'''
基于字典的字符串格式化
并且标记也不是在字符串中的一个简单 %s ,而是包含
了一个用括号包围起来的名字。这个名字是 params dictionary 中的一个键
字,所以 %(pwd)s 标记被替换成相应的值
'''

def foo(args):
    '命名空间配合 字典的字符串格式化，效果甚好'
    x = 10
    print('arg is %(args)s, variate is %(x)s' % locals())


def my_generator():
    i = 1
    while i < 1000:

        i+=1
        yield i


if __name__ == '__main__':
    dict1 = {'user': 'mars', 'pwd': '1234', 'id': '001'}
    print('user is %(user)s, id is %(id)s, password is %(pwd)s'%dict1)
    foo(100)
    sys = __import__('sys')

    for i in range(100):
        print(my_generator().__next__())