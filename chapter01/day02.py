#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day02.py
@time: 2019/03/03
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import string
import stat
import os


def info(object, spacing=10, collapse=1):
    """
    print methods and doc strings
    Takes module , class, list, dictionary, or string.
    """
    methodlist = [method for method in dir(object) if callable(getattr(
        object, method
    ))]

    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s : s)
    print("\n".join(['%s %s' % (method.ljust(spacing), processFunc(
        str(getattr(object, method).__doc__)
    ))for method in methodlist]))

'''
利用getattr 创建分发者，可以为每种的输出格式定义各自的格式输出函数
'''
def output(data, format='text'):
    '利用getattr 创建分发者，可以为每种的输出格式定义各自的格式输出函数'
    output_function = getattr(stat, 'output_%s' % format)
    return output_function(data)

def demo1():
    s = ['li', 'mars', 'tom', 'john']

    new_string = ';'.join(s)

    new_list = new_string.split(';', -1)
    print(new_list)
    # dir 返回任意对象的属性，方法列表
    print(dir(new_list).__doc__)
    # callable()判断参数对象是否可调用
    print(callable(new_list))
    print(callable(info))
    print(s)

    # getattr()的使用,可以调用对象的任何属性和方法相当于  object.y  /// object.y()
    pop_one = getattr(s, 'pop')()
    # s.pop()
    print(pop_one)
    print(s)
    getattr(s, 'append')('mark')
    print(s)
    getattr(s, 'clear')()
    print(s)

def boolean_user():
    # and or boolean逻辑运算符并不返回boolean 而是返回 具体使条件成立的值
    print('a' and 'b')
    print('' and 'b')
    # lamble 表达式不需要ｒｅｔｕｒｎ
    y = lambda x: x if x < 10 else x + 12
    print(y(30))
    # repr 返回一个对象的字符串表示
    print(repr(output))

if __name__ == '__main__':

    print('\n'.join(['yes', 'no', 'no-yes']))

    print(os.path)
    #  os.path.split 对路径的切割有着特殊的作用，但是注意 unix和window的路径的命名的方式的不同
    (path, file) = os.path.split('c:/music/ap/mahadeva.mp3')
    print(path)
    print(file)
    print(os.path.splitext(file))

    try:
        f = open('/home/mars/Data/000032.csv', 'rb')
        print(f.mode)
        print(f.name)
        # 获得文件访问指针的位置
        print(f.tell())
        f.seek(128, 1)
        print(f.tell())
        # 移动对应的文件的位置（以字节为单位）
        f.seek(128,2)
        print(f.tell())
    except IOError:
        print('error io')
    finally:
        if not f.closed:
            print(f.closed)
            print('close the file')
            # 文件已经关闭的文件对象调用 close 不会 引发异常,它静静地失败
            f.close()

    dir_list = os.listdir('/home/mars')
    print('\n'.join(dir_list))







