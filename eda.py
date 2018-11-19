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
question=pd.read_csv('input/question_id.csv')
train=pd.read_csv('input/train.csv')
print(train['label'].value_counts())
print(train['qid1'].value_counts()[:10]) # 问题有重复的

# 句子长度统计
question['word_len']=question['wid'].apply(lambda x:len(x.split(' ')))
question['char_len']=question['cid'].apply(lambda x:len(x.split(' ')))
question['word_len'].value_counts()[:40].plot(kind='bar')
plt.show()
print(question.describe())

# 词频统计
words=[]
for word_sent in question['wid']:
    words.extend(word_sent.split(' '))
print(Counter(words),len(Counter(words)))

words=[]
for word_sent in question['cid']:
    words.extend(word_sent.split(' '))
print(Counter(words),len(Counter(words)))