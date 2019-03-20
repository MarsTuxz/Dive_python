#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: __init__.py.py
@time: 2019/03/02
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd


from sklearn.datasets import fetch_mldata
from sklearn import datasets
import numpy as np

mnist = fetch_mldata('mnist-original', data_home='/home/mars/Data')
print(mnist.DESCR)
print(mnist.data)
print(mnist.COL_NAMES)
