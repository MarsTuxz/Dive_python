#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day03.py
@time: 2019/03/23
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import voting_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

'''
集成学习
如果所有分类器都能够估算出类别的概率(即有
predict_proba()方法),那么你可以将概率在所有单个分类器上平
均,然后让Scikit-Learn给出平均概率最高的类别作为预测。这被称为
软投票法.

默认情况
下,SVC类是不行的,所以你需要将其超参数probability设置为
True(这会导致SVC使用交叉验证来估算类别概率,减慢训练速度,
并会添加predict_proba()方法)
'''
def ensemble_voting():
    log_clf = LogisticRegression()
    rnd_clf = RandomForestClassifier()
    svm_clf = SVC()
    voting_clf = voting_classifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard'
    )
    #　查看每个分类器在测试集上的准确率
    from sklearn.metrics import accuracy_score
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit()
        y_pred = clf.predict()
        print(clf.__class__.__name__, accuracy_score())

'''
每个训练器使用的算法相同，但是在不同的训练集上进行驯良。
bagging是有放回的抽样　pasting是有无放回的抽样

BaggingRegressor用于回归
这是一个bagging的示例,如果你
想使用pasting,只需要设置bootstrap=False即可

如果基础分类器能够估算类别概率(也就是具备
predict_proba()方法),比如决策树分类器,那么BaggingClassifier
自动执行的就是软投票法而不是硬投票法
'''
def ensemble_bagging_pasting():
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1,
        oob_score=True
    )
    bag_clf.fit()
    y_pred = bag_clf.predict()
    '''
    当使用ｂａｇｇｉｎｇ的时候大约有３７％　的数据在训练集中不会被使用，所以，可以直接使用这些
    包外的数据进行测试。
    创建BaggingClassifier时,设置
    oob_score=True,就可以请求在训练结束后自动进行包外评估
    '''
    bag_clf.oob_score_
    # 每个训练实例的包外决策函数也可以通过变量
    # oob_decision_function_获得,将以数组的形式返回评估的结果

    '''
    BaggingClassifier也支持对特征进行抽样,这通过两个超参数控
制:max_features和bootstrap_features。它们的工作方式跟
max_samples和bootstrap相同,只是抽样对象不再是实例,而是特
征。因此,每个预测器将用输入特征的随机子集进行训练。
这对于处理高维输入(例如图像)特别有用。对训练实例和特征
都进行抽样,被称为Random
Patches方法。 [1] 而保留所有训练实例
(即bootstrap=False并且max_samples=1.0)但是对特征进行抽样(即
bootstrap_features=True并且/或max_features<1.0),这被称为随机子
空间法。 [2]
    '''




'''
adabost
当只有两个类别时,SAMME即等同于AdaBoost。此外,
如果预测器可以估算类别概率(即具有predict_proba()方法),
Scikit-Learn会使用一种SAMME的变体,称为SAMME.R(R代
表“Real”),它依赖的是类别概率而不是类别预测,通常表现更好
'''
def adabost_mars():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=5), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5
    )
    '''
    如果你的AdaBoost集成过度拟合训练集,你可以试试减少估
    算器数量,或是提高基础估算器的正则化程度
    '''
    ada_clf.fit()




'''
Gradient Boosting
不同之处在于,它不是
像AdaBoost那样在每个迭代中调整实例权重,而是让新的预测器针对
前一个预测器的残差进行拟合
主要是针对决策树
'''
def gragoost():
    from sklearn.ensemble import GradientBoostingRegressor
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
    gbrt.fit()

if __name__ == '__main__':
    pass
