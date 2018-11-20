# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: eda.py 
@Time: 2018/11/19 14:12
@Software: PyCharm 
@Description:
"""
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

question = pd.read_csv('input/question_id.csv')
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# train.rename(columns={'qid1':'q1','qid2':'q2'},inplace=True)
# test.rename(columns={'qid1':'q1','qid2':'q2'},inplace=True)
# question.rename(columns={'wid':'words','cid':'chars'},inplace=True)
#
# train.to_csv('train.csv',index=False)
# test.to_csv('test.csv',index=False)
# question.to_csv('question.csv',index=False)

print(train['label'].value_counts())
print(train['qid1'].value_counts()[:10])  # 问题有重复的

# 句子长度统计
question['word_len'] = question['wid'].apply(lambda x: len(x.split(' ')))
question['char_len'] = question['cid'].apply(lambda x: len(x.split(' ')))
question['word_len'].value_counts()[:40].plot(kind='bar')
plt.show()
print(question.describe())

# 词频统计
words = []
for word_sent in question['wid']:
    words.extend(word_sent.split(' '))
print(Counter(words), '\n', len(Counter(words)))

words = []
for word_sent in question['cid']:
    words.extend(word_sent.split(' '))
print(Counter(words), '\n', len(Counter(words)))

import numpy as np
a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
b = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
c = np.stack((a, b), axis=0)
print(c[:,:,:])
print("结果一样")
print(c)
print("---")

print(c[:1,:,:])



