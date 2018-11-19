# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: Char_Multi_LSTM_CNN.py
@Time: 2018/11/14 17:11
@Software: PyCharm 
@Description:
"""

import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Concatenate, Dropout, BatchNormalization, GRU, LSTM, Conv1D, \
    MaxPool1D, Flatten, Lambda, merge,concatenate
from keras.layers.wrappers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
que = pd.read_csv('input/question_id.csv')


# 读入词向量文件
def embed_dict(file):
    temp = {}
    with open(file) as f:
        for line in f.readlines():
            s = line.strip('\n').split('\t')
            temp[s[0]] = [float(v) for v in s[1:]]
    return temp


# 读入train和test
def read_data(typein, data):
    data = pd.merge(data, que[['qid', 'cid']], left_on='qid1', right_on='qid', how='left')
    data = pd.merge(data, que[['qid', 'cid']], left_on='qid2', right_on='qid', how='left')
    data.drop(['qid_x', 'qid_y'], axis=1, inplace=True)
    data.to_csv('demo.csv', index=None)

    if typein == 'train':
        columns = ['qid1', 'qid2','label', 'word1', 'word2']
    else:
        columns = ['qid1', 'qid2', 'label','word1', 'word2']
    data.columns = columns

    return data


# texts_to_sequences
def text2seq(q1, q2, tokenizer, MSL=25):
    return pad_sequences(tokenizer.texts_to_sequences(q1), maxlen=MSL), pad_sequences(tokenizer.texts_to_sequences(q2),
                                                                                      maxlen=MSL)


# 构建embedding矩阵
def embedding_matrix(w_inx, w_dict, MAX_NB_WORDS, EMBEDDING_DIM):
    word_embedding_matrix = np.zeros((MAX_NB_WORDS + 1, EMBEDDING_DIM))
    for word, i in w_inx.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = w_dict.get(str(word).upper())
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector
    return word_embedding_matrix

#全局变量
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 300
#######################
word_dict = embed_dict('input/char_embedding.txt')
test = read_data('test',test)
train = read_data('train',train)
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(que['cid'])

word_index = tokenizer.word_index
q1_data_tr,q2_data_tr = text2seq(train['word1'],train['word2'],tokenizer)
q1_data_te,q2_data_te = text2seq(test['word1'],test['word2'],tokenizer)
q_concat = np.stack([q1_data_tr,q2_data_tr],axis=1)
word_embedding_matrix = embedding_matrix(word_index,word_dict,MAX_NB_WORDS, EMBEDDING_DIM)


def lstm_layer(q, lstm1, lstm2):
    q = lstm1(q)
    q = Dropout(0.3)(q)
    q = lstm2(q)
    q = Lambda(lambda x: K.reshape(x, (-1, 25, 256)))(q)
    return q


def conv_pool(conv_unit, q):
    q_conv = conv_unit(q)
    q_maxp = MaxPool1D(pool_size=25)(q_conv)
    q_maxp = Lambda(lambda x: K.reshape(x, (-1, int(x.shape[-1]))))(q_maxp)
    q_meanp = Lambda(lambda x: K.mean(x, axis=1))(q_conv)
    return q_maxp, q_meanp


def mix_layer(q1_maxp, q1_meanp, q2_maxp, q2_meanp):
    add_q_max = Lambda(lambda x: x[0] + x[1])([q1_maxp, q2_maxp])
    sub_q_max = Lambda(lambda x: K.abs(x[0] - x[1]))([q1_maxp, q2_maxp])
    mul_q_max = concatenate([q1_maxp, q2_maxp])
    square_max = Lambda(lambda x: K.square(x[0] - x[1]))([q1_maxp, q2_maxp])

    add_q_mean = Lambda(lambda x: x[0] + x[1])([q1_meanp, q2_meanp])
    sub_q_mean = Lambda(lambda x: K.abs(x[0] - x[1]))([q1_meanp, q2_meanp])
    mul_q_mean = concatenate([q1_meanp, q2_meanp])
    square_mean = Lambda(lambda x: K.square(x[0] - x[1]))([q1_meanp, q2_meanp])

    return Concatenate()([q1_maxp, q2_maxp, add_q_max, sub_q_max, mul_q_max, square_max,
                          q1_meanp, q2_meanp, add_q_mean, sub_q_mean, mul_q_mean, square_mean])

re = []
from sklearn.model_selection import StratifiedKFold
for i,(tr,va) in enumerate(StratifiedKFold(n_splits=10).split(q_concat,train['label'].values)):
    Q1_train = q_concat[tr][:,0];Q2_train = q_concat[tr][:,1]
    Q1_test = q_concat[va][:,0];Q2_test = q_concat[va][:,1]
    #构建embedding层，q1 和 q2共享此embedding层
    embedding_layer = Embedding(MAX_NB_WORDS+1,EMBEDDING_DIM,weights=[word_embedding_matrix],input_length=25,trainable=False)
    #词嵌入
    sequence_1_input = Input(shape=(25,), dtype='int32')
    embed_1 = embedding_layer(sequence_1_input)
    sequence_2_input = Input(shape=(25,), dtype='int32')
    embed_2 = embedding_layer(sequence_2_input)
    #lstm
    lstm_layer_1 = LSTM(256,return_sequences=True)
    lstm_layer_2 = LSTM(256,return_sequences=True)
    q1 = lstm_layer(embed_1,lstm_layer_1,lstm_layer_2)
    q2 = lstm_layer(embed_2,lstm_layer_1,lstm_layer_2)
    #用类似TextCNN的思路构建不同卷积核的特征，两个句子共用同样的卷积层
    kernel_size = [2,3,4,5]
    conv_concat = []
    for kernel in kernel_size:
        conv = Conv1D(64,kernel_size=kernel,activation='relu',padding='same')
        q1_maxp,q1_meanp = conv_pool(conv,q1)
        q2_maxp,q2_meanp = conv_pool(conv,q2)
        mix = mix_layer(q1_maxp,q1_meanp,q2_maxp,q2_meanp)
        conv_concat.append(mix)
    conv = Concatenate()(conv_concat)
    #全连接层
    merged = Dropout(0.3)(conv)
    merged = BatchNormalization()(merged)
    merged = Dense(512, activation='relu',name='dense_output')(merged)
    merged = Dropout(0.3)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(256, activation='relu',name='dense_output2')(merged)
    merged = Dropout(0.3)(merged)
    merged = BatchNormalization(name='bn_output')(merged)
    preds = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[sequence_1_input, sequence_2_input],outputs=preds)
    model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['acc'])
    hist = model.fit([Q1_train, Q2_train], train['label'].values[tr],
                 validation_data=([Q1_test, Q2_test], train['label'].values[va]),
                 epochs=50,
                 batch_size=1024,
                 shuffle=True,
                 callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=5,mode='min')])
    pred = model.predict([q1_data_te,q2_data_te],batch_size=1024)
    avg = [v[0] for v in pred]
    re.append(avg)


avg = np.mean(re,axis=0)
preds=[]
for p in avg:
    if p>=0.5:
        preds.append(1)
    else:
        preds.append(0)
test['label']=preds
test[['qid1','qid2','label']].to_csv('result/01_lgb_cv5.csv',columns=['qid1','qid2','label'], index=None)