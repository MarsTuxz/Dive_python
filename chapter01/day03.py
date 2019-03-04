#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day03.py
@time: 2019/03/03
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import re

if __name__ == '__main__':
    s = '100 road oroad'
    # $ 表示以只在末尾替换
    new_s = re.sub('road$', 'rd.', s)
    print(s)
    print(new_s)
    # \b 表示边界，也就是说，匹配一个完整的单词
    new_s1 = re.sub(r'\broad\b', 'rd.', s)
    print(new_s1)

    # ?表示即可匹配也可不匹配, ^ 表示从开头匹配
    pattern = '^mm?m?$'
    print(re.search(pattern, 'm'))
    print(re.search(pattern, 'mmm'))
    print(re.search(pattern, 'mmmm'))
    # {}表示匹配的次数的区间
    pattern1= '^m{0,4}$'
    print(re.search(pattern1, 'mmmm'))
    print(re.search(pattern1, 'mmm'))
    #
    pattern2 = '^mm(m|n|f)$'
    print(re.search(pattern2, 'mmmm'))
    print(re.search(pattern2, 'mmf'))
    print(re.search(pattern2, 'mmn'))
    re.findall()
