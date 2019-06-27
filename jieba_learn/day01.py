#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:mars
@file: day01.py
@time: 2019/05/11
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jieba
import jieba.analyse
import jieba_learn.jieba_get_keyword as get_keyword
import re
'''
jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=()) 
sentence 为待提取的文本
topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
withWeight 为是否一并返回关键词权重值，默认值为 False
allowPOS 仅包括指定词性的词，默认值为空，即不筛选
jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件
'''


# 创建停用词list
def stopwordslist(stopwords_filepath):
    stopwords = [line.strip() for line in open(stopwords_filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 对句子进行分词
def segment(text, stopwords):
    # 去掉英文
    text = re.sub('[a-zA-Z]', '', text)
    text = re.sub('[0-9]', '', text)
    seg_list_without_stopwords = []
    seg_list = jieba.cut(text)
    for word in seg_list:
        if word not in stopwords:
            if word != '\t':
                seg_list_without_stopwords.append(word)
    return seg_list_without_stopwords

if __name__ == '__main__':
    sentence = '妹妹超级喜欢年糕火锅，每回放假回来都得吃'
    texts = []
    stopwords = stopwordslist('哈工大停用词表.txt')
    words = segment(sentence, stopwords)
    word = ''.join(words)
    print(word)


    seg_list = jieba.cut_for_search(sentence)
    print(" ".join(seg_list))
    tags = jieba.analyse.extract_tags(sentence, topK=10)
    print(tags)
    t_tags = jieba.analyse.extract_tags(word, topK=10,allowPOS=('Ag', 'ad', 'a', 'an','d', 'n','vn','ns', 'v','nr','c'))
    print(t_tags)

    '''
    自定义关键词提取
    '''
    print('自定义关键词提取')
    self_tag = get_keyword.textrank(word,topK=10, allowPOS=('Ag', 'ad', 'a', 'an','d', 'n','vn','ns', 'v','nr','c'))
    print(self_tag)