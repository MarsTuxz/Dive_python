#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day05.py
@time: 2019/03/13
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def operation():
    '''
        数组的除法
        '''
    a = np.array([2, 3, 5])
    b = np.array([12, 13, 15])
    print('Divide :', np.divide(a, b))
    print('true_divide :', np.true_divide(b, a))
    print('floor_divide :', np.floor_divide(b, a))
    '''
    数组的模运算
    '''
    # remainder 函数逐个返回两个数组中元素相除后的余数。
    # 如果第二个数字为0,则直接
    # % 操作符仅仅是remainder函数的简写
    # 返回0:
    print('remainder:', np.remainder(b, a))
    print('remainder:', np.remainder(b, 4))

    # fmod 函数处理负数的方式与 remainder 、 mod 和 % 不同。所得余数的正负由被除数决定,
    # 与除数的正负无关
    print('fmod:', np.fmod([3, -5, 6], -4))

    # 小的玩意
    a = float(9)
    b = float(8)
    t = np.linspace(-np.pi, np.pi, 201)
    x = np.sin(a * t + np.pi / 2)
    y = np.sin(b * t)
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    '''
    求解线性方程组
    '''
    #创建一个矩阵
    A = np.mat("1 -2 1;0 2 -8;-4 5 9")
    print("A\n", A)
    b = np.array([0, 8, -9])
    print("b\n", b)
    # solve 求解线性方程组
    x = np.linalg.solve(A, b)
    print(x)
    # 特征值和特征向量
    A = np.mat('3, -2;1, 0')
    print(A)
    # 调用eigvals函数求解特征值
    print("Eigenvalues:", np.linalg.eigvals(A))
    # 使用 eig 函数求解特征值和特征向量。该函数将返回一个元组,按列排放着特征值和对
    # 应的特征向量,其中第一列为特征值,第二列为特征向量。
    print("eigenvector / Eigenvalues:", np.linalg.eig(A))
    # 奇异值分解
    A = np.mat("4 11 14;8 7 -2")
    print("A\n", A)
    U, Sigma, V = np.linalg.svd(A, full_matrices=False)
    # 使用 pinv 函数计算广义逆矩阵:
    # 使用 det 函数计算行列式:
    # 计算傅里叶变换
    #np.all()
    # np.random.binomial 函数 模拟随机游走
    # np.random.hypergeometric 函数 模拟超几何分布
    # np.random.normal函数模拟正太分布
    pd.DataFrame().to_csv()