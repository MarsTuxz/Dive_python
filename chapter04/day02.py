#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day02.py
@time: 2019/03/23
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split ,StratifiedShuffleSplit


'''
回归模型的学习曲线的绘制
如果曲线是典型的模型拟合不足。两条曲线均到达高地,非
常接近,而且相当高

如果两条曲线之间有一定差距。这意味着该模型在训练数据上的表
现比验证集上要好很多,这正是过度拟合的标志
'''
def plot_learning_curves(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_errors, test_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        test_errors.append(mean_squared_error(y_test_predict, y_test))
        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
        plt.plot(np.sqrt(test_errors), "b-", linewidth=3, label="val")


'''
通过早起停止算法，来达到防止过多次的随机梯度的下降的计算
'''
def early_stop(X_train_poly_scaled, y_train, X_val_poly_scaled, y_val):
    from sklearn.base import clone
    from sklearn.linear_model import SGDRegressor
    sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None, learning_rate="constant")
    minimum_val_error = float("inf")
    best_epoch = None
    best_model = None
    for epoch in range(1000):
        sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off
        y_val_predict = sgd_reg.predict(X_val_poly_scaled)
        val_error = mean_squared_error(y_val_predict, y_val)
        if val_error < minimum_val_error:
            minimum_val_error = val_error
            best_epoch = epoch
            best_model = clone(sgd_reg)

'''
SVM
'''
if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica
    print(list(y).count(0))
    print(list(y).count(1))
    stf = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=45)
    for train_index, test_index in stf.split(X, y):
        train_data_x = X[train_index]
        test_data_x = X[test_index]
        train_data_y = y[train_index]
        test_data_y = y[test_index]
        print(train_data_x.shape)

    svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge")),
    ))
    svm_clf.fit(train_data_x, train_data_y)
    predict_y = svm_clf.predict([[5.5, 1.7]])
    print(predict_y)
    score1 = svm_clf.score(test_data_x, test_data_y)
    print(score1)

    '''
    直接添加多项式特征
    '''
    print('添加特征')
    from sklearn.datasets import make_moons
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    # 添加多项式特征
    polynomial_svm_clf = Pipeline((
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=1, loss="hinge"))
    ))
    polynomial_svm_clf.fit(train_data_x, train_data_y)
    score2 = polynomial_svm_clf.score(test_data_x, test_data_y)
    print(score2)
    '''
    核函数添加多项式特征
    '''
    from sklearn.svm import SVC
    print('核函数添加特征')
    poly_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=1))
    ))
    poly_kernel_svm_clf.fit(train_data_x, train_data_y)
    score3 = poly_kernel_svm_clf.score(test_data_x, test_data_y)
    print(score3)
    '''
    核函数添加相似特征（高斯核）
    一般使用高斯核，会将原始的特征映射到更高维的空间（）
    这会创造出许多维度,因而也增加了转
    换后的训练集线性可分离的机会。缺点是,一个有m个实例n个特征
    的训练集会被转换成一个m个实例m个特征的训练集(假设抛弃了原
    始特征)
    '''

    rbf_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ))
    rbf_kernel_svm_clf.fit(train_data_x, train_data_y)
    '''
    有这么多的核函数,该如何决定使用哪一个呢?有一个经验
    法则是,永远先从线性核函数开始尝试(要记住,LinearSVC比
    SVC(kernel="linear")快得多),特别是训练集非常大或特征非常多
    的时候。如果训练集不太大,你可以试试高斯RBF核,大多数情况下
    它都非常好用
    '''