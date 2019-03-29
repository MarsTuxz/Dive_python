#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day01.py
@time: 2019/03/22
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

'''
对数据进行分层抽样
'''
def StratifiedShuffleSplit_mars():
    data = pd.read_csv('000032.csv')
    # 统计各个类别分类的数目
    # print(data['close'].value_counts())
    # 　各个类别的统计信息
    print(data.info())
    print(data.describe())
    # 绘制各个维度的直方图
    # data.hist(bins=50, figsize=(20, 15))
    # plt.show()
    # 将ｃｌｏse 类别取整
    data['other'] = np.ceil(data['close'])
    print(data['other'].value_counts())
    # 大于１８的合并为１８
    data['other'].where(data.other < 18.0, 18, inplace=True)
    print(data['other'].value_counts())
    # print(data.info())

    '''
    分层抽样(对ｏｔｈｅｒ进行分层抽样)
    '''
    stf = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=45)
    for train_index, test_index in stf.split(data, data['other']):
        train_data = data.loc[train_index]
        test_data = data.loc[test_index]
        print(train_index)
        print('***')
    print(data.other.value_counts() / len(data))
    print(train_data.other.value_counts() / len(train_data))
    print(test_data.other.value_counts() / len(test_data))
    print('随机抽样')
    train_ram, test_ram = train_test_split(data, test_size=0.2, random_state=45)
    print(train_ram.other.value_counts() / len(train_ram))
    print(test_ram.other.value_counts() / len(test_ram))

'''
探究各个维度数据与目标的含义
'''
def get_info():
    data = pd.read_csv('000032.csv')
    # 探究各个维度的数据隐含的含义
    data['other'] = np.ceil(data['volume'] / 10000)
    data['other'].where(data['other'] < 10, 10, inplace=True)
    data['other'].where(data['other'] > 2, 2, inplace=True)
    print(data['other'].value_counts())
    data.plot(kind='scatter', x='other', y='close', alpha=0.2)

    data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
              s=data["open"] / 100, label="population",
              c="low", cmap=plt.get_cmap("jet"), colorbar=True,
              )
    plt.legend()

    plt.show()


'''
文本标签转数字
'''
def label_Encoder():
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    #文本特征
    string = data['string']
    string_encoded = encoder.fit_transform(string)
    # 映射表
    print(encoder.classes_)


'''
保存模型
'''
def save_model(model):
    from sklearn.externals import joblib
    from sklearn.model_selection import RandomizedSearchCV
    # 随机搜索模型
    joblib.dump(model, 'my_model.pkl')
    # 导入模型
    rd_model = joblib.load('my_model.pkl')


'''
缺失值填充
'''
def fill_na():
    '''
        缺失值填充
        '''
    data = pd.read_csv('000032.csv', index_col='date')
    from sklearn.preprocessing import Imputer
    print(data[:4])
    # data.drop('date', axis=0)
    imputer = Imputer(strategy='median')
    imputer.fit_transform(data)
    # 各个维度的中位数
    print(imputer.statistics_)

'''
利用手写字体做分类(随机梯度下降)
在这个例子中，我要学习到，怎么对分类器的结果，进行模型的分析(精确率，召回率)，
从而改进模型
'''
def predict_mnist_for_SGD():
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('mnist-original', data_home='/home/mars/Data')
    #print(mnist)
    x, y = mnist['data'], mnist.target
    print(x.shape, y.shape)
    #shuffle_index = np.random.permutation(60000)
    #X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # 变成一个２分类问题
    y_train_5 = (y_train == 5)
    print(y_train_5.shape)
    y_test_5 = (y_test == 5)
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(random_state=42)
    '''
    均衡抽样，交叉验证(3折)
    '''
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone
    skfolds = StratifiedKFold(n_splits=3, random_state=42)
    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = (y_train_5[train_index])
        X_test_fold = X_train[test_index]
        y_test_fold = (y_train_5[test_index])
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))  # prints 0.9502, 0.96565 and 0.96495


    '''
    非均衡抽样
    '''

    sgd_clf.fit(X_train, y_train_5)
    #print(sgd_clf.predict(X_test))
    # 评估性能
    from sklearn.model_selection import cross_val_score, cross_val_predict
    score = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    print(score)
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    from sklearn.metrics import confusion_matrix
    #混淆矩阵
    cm = confusion_matrix(y_train_5, y_train_pred)
    print(cm)
    from sklearn.metrics import precision_score, recall_score, f1_score
    f1_score = f1_score(y_train_5, y_train_pred)
    print(f1_score)
    # 通过调整决策的阀值在精度和召回率之间做选择
    some_digit = X_train[39000]
    print(some_digit.shape)
    '''
    # 返回的是一个决策的阀值
    注意　RandomForestClassifier类没有decision_function()方法,相反,它有
    的是dict_proba()方法
    y_scores = sgd_clf.decision_function(X_train[:10])
    print(sgd_clf.predict(X_train[:10]))
    print(np.mean(y_scores))
    # 调整阀值,进行预测
    y_some_digit_pred = (y_scores > np.mean(y_scores))
    print(y_some_digit_pred)
    '''
    # 确实最优的阀值,要它返回的是决策分数
    # cross_val_predict()通过交叉验证，返回决策分数(threshold)
    y_decision_score = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
    method="decision_function")
    # 使用precision_recall_curve计算，所有的决策分数，对应的精确率和召回率
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_decision_score)
    print(len(thresholds))
    print(len(y_decision_score))
    '''
    绘制，决策分数，召回率，精确率图
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.plot(thresholds, precisions[:-1], "b--", label="Precision")
    ax1.plot(thresholds, recalls[:-1], "g-", label="Recall")
    #ax1.xlabel("Threshold")
    ax1.legend(loc="upper left")
    #ax1.y([0, 1])


    '''
    精确度和召回率 曲线
    '''
    ax2.plot(precisions, recalls, 'r', label="Precision_Recall_")
    ax2.legend(loc="upper left")
    plt.ylim([0, 1])

    '''
    ROC曲线
    ｘ : FPR是被错误分为正类的负类实例比率(假正类率)
    y: 召回率（所有实际正类中，预测正确的比率）（ＴＰＲ）
    一个优秀的分类器应该离这条线越远越好(向左上角)
    
    有一种比较分类器的方法是测量曲线下面积(AUC)。完美的
    分类器的ROC AUC等于1,而纯随机分类器的ROC AUC等于0.5。
    '''
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_train_5, y_decision_score)
    print('auc:', auc)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_train_5, y_decision_score)
    ax3.plot(fpr, tpr, linewidth=2, label='roc')
    ax3.plot([0, 1], [0, 1], 'k--')
    ax3.axis([0, 1, 0, 1])
    ax3.legend()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    plt.show()
    # 调整决策的阀值
    # 需要高的，准确性
    y_train_pred_90 = (y_decision_score > 210000)
    print('精确度：', precision_score(y_train_5, y_train_pred_90))
    print('召回率:',recall_score(y_train_5, y_train_pred_90))

'''
利用手写字体做分类(随机森林)
在这个例子中，我要学习到，怎么对分类器的结果，进行模型的分析(精确率，召回率)，
从而改进模型
'''

def predict_mnist_for_RF():
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('mnist-original', data_home='/home/mars/Data')
    #print(mnist)
    x, y = mnist['data'], mnist.target
    print(x.shape, y.shape)
    #shuffle_index = np.random.permutation(60000)
    #X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    '''
    '''
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # 变成一个２分类问题
    y_train_5 = (y_train == 5)
    print(y_train_5.shape)
    y_test_5 = (y_test == 5)
    from sklearn.ensemble import RandomForestClassifier
    forest_clf = RandomForestClassifier(random_state=42)
    from sklearn.model_selection import cross_val_score, cross_val_predict
    #预测决策的概率
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                        method="predict_proba")
    forest_clf.fit(X_train, y_train_5)
    forest_clf.predict(X_train)
    print(y_probas_forest[:5])
    from sklearn.metrics import roc_curve, roc_auc_score
    y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
    '''
    绘制，ｒｏｃ曲线，　和对应的auc 面积
    '''
    auc = roc_auc_score(y_train_5, y_scores_forest)
    print('auc:', auc)
    plt.plot(fpr_forest, tpr_forest, linewidth=2, label='roc')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.legend()
    plt.legend(loc="bottom right")
    plt.show()
    save_model(forest_clf)



if __name__ == '__main__':
    predict_mnist_for_RF()