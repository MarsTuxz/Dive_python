#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: stacking_mars.py
@time: 2019/03/23
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd


'''
堆叠法进行模型的集成
与其使用一些简单的函数(比如硬投票)来聚合集成中所有预测
器的预测,我们为什么不训练一个模型来执行这个聚合呢?图7-12显
示了在新实例上执行回归任务的这样一个集成。底部的三个预测器分
别预测了不同的值(3.1、2.7和2.9),然后最终的预测器(称为混合
器或元学习器)将这些预测作为输入,进行最终预测
'''



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from brew.base import Ensemble, EnsembleClassifier
from brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
from brew.combination.combiner import Combiner

from mlxtend.data import iris_data
#from mlxtend.evaluate import plot_decision_regions

'''
代码学习链接　　https://github.com/viisar/brew

在需要的时候可以学学人家怎么写的这些代码。原理可以公用
'''
if __name__ == '__main__':
    # Initializing Classifiers
    clf1 = LogisticRegression(random_state=0)
    clf2 = RandomForestClassifier(random_state=0)
    clf3 = SVC(random_state=0, probability=True)

    # Creating Ensemble
    ensemble = Ensemble([clf1, clf2, clf3])
    eclf = EnsembleClassifier(ensemble=ensemble, combiner=Combiner('mean'))

    # Creating Stacking
    layer_1 = Ensemble([clf1, clf2, clf3])
    layer_2 = Ensemble([sklearn.clone(clf1)])

    stack = EnsembleStack(cv=3)

    stack.add_layer(layer_1)
    stack.add_layer(layer_2)

    sclf = EnsembleStackClassifier(stack)

    clf_list = [clf1, clf2, clf3, eclf, sclf]
    lbl_list = ['Logistic Regression', 'Random Forest', 'RBF kernel SVM', 'Ensemble', 'Stacking']

    # Loading some example data
    X, y = iris_data()
    X = X[:,[0, 2]]

    # WARNING, WARNING, WARNING
    # brew requires classes from 0 to N, no skipping allowed
    d = {yi : i for i, yi in enumerate(set(y))}
    y = np.array([d[yi] for yi in y])

    # Plotting Decision Regions
    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize=(10, 8))

    itt = itertools.product([0, 1, 2], repeat=2)

    # for clf, lab, grd in zip(clf_list, lbl_list, itt):
    #     clf.fit(X, y)
    #     # ax = plt.subplot(gs[grd[0], grd[1]])
    #     # fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    #     # plt.title(lab)
    # #plt.show()

    for clf in clf_list:
        clf.fit(X, y)

