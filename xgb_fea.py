# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: xgb_fea.py
@Time: 2018/11/14 14:04
@Software: PyCharm 
@Description:
"""
import pandas as pd
import numpy as np
import math
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from fuzzywuzzy import fuzz # 字符串模糊匹配工具 http://hao.jobbole.com/fuzzywuzzy/
from gensim.models.doc2vec import Doc2Vec,LabeledSentence

from sklearn.model_selection import KFold,RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import PolynomialFeatures


char_embedding=pd.read_csv('input/char_embedding.txt',sep='\t',header=None) # 300维 2307个字
# print(dict(zip(char_embedding[0],char_embedding.loc[:,1:].values.tolist()))['C100000'])
word_embedding=pd.read_csv('input/word_embedding.txt',sep='\t',header=None) # 300维 9647个词

df_train=pd.read_csv('input/train.csv')
df_test=pd.read_csv('input/test.csv')
df_questions=pd.read_csv('input/question_id.csv')
print(len(df_train),len(df_test),len(df_questions))

df_train=pd.merge(df_train,df_questions,left_on='qid1',right_on='qid',how='left') # 问题1
df_train.rename(columns={'wid':'wid1','cid':'cid1'},inplace=True)
df_train=pd.merge(df_train,df_questions,left_on='qid2',right_on='qid',how='left') # 问题2
df_train.rename(columns={'wid':'wid2','cid':'cid2'},inplace=True)

df_test=pd.merge(df_test,df_questions,left_on='qid1',right_on='qid',how='left')
df_test.rename(columns={'wid':'wid1','cid':'cid1'},inplace=True)
df_test=pd.merge(df_test,df_questions,left_on='qid2',right_on='qid',how='left')
df_test.rename(columns={'wid':'wid2','cid':'cid2'},inplace=True)

df_all=pd.concat([df_train,df_test],axis=0,ignore_index=True)
df_all.drop(columns=['qid_x','qid_y'],inplace=True)

train_len = len(df_train)


def train_doc2vec():
    print("train doc2vec...")
    tag_tokenized=[gensim.models.doc2vec.TaggedDocument(text.split(' '),[qid]) for qid,text
                   in zip(df_questions['qid'],df_questions['wid'])]

    model=Doc2Vec(size=300,min_count=1,iter=200)
    model.build_vocab(tag_tokenized)
    model.train(tag_tokenized,total_examples=model.corpus_count,epochs=model.iter)
    model.save('model/word_doc2vec.model')

    print("done.")
# train_doc2vec()


def Cosine(vec1, vec2):
    """
    余弦相似度
    :param vec1:
    :param vec2:
    :return:
    """
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return npvec1.dot(npvec2)/(math.sqrt((npvec1**2).sum()) * math.sqrt((npvec2**2).sum()))


def Euclidean(vec1, vec2):
    """
    欧氏距离
    :param vec1:
    :param vec2:
    :return:
    """
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((npvec1-npvec2)**2).sum())


def calculate_sim():
    """
    计算相似度
    :return:
    """
    print("get doc sim...")
    doc_sims=[]
    cos_sim=[]
    eu_dis=[]
    model_dm=Doc2Vec.load('model/word_doc2vec.model')
    for qid1,qid2 in tqdm(zip(df_all['qid1'],df_all['qid2'])):
        doc_sims.append(model_dm.docvecs.similarity(qid1,qid2)) # 参数文档对应的id
        cos_sim.append(Cosine(model_dm.docvecs[qid1],model_dm.docvecs[qid2]))
        eu_dis.append(Euclidean(model_dm.docvecs[qid1],model_dm.docvecs[qid2]))
    return doc_sims,cos_sim,eu_dis


def get_fuzzy_ratios():
    print("get fuzz ratios...")
    ratios=[] # 简单比
    for wid1,wid2 in tqdm(zip(df_all['wid1'],df_all['wid2'])):
        ratios.append(fuzz.ratio(wid1,wid2))
    return ratios


doc_sims,cos_sim,eu_dis=calculate_sim()
ratios=get_fuzzy_ratios()

df_all['doc_sims']=doc_sims
df_all['cos_sim']=cos_sim
df_all['eu_dis']=eu_dis
df_all['ratios']=ratios


def add_poly_features(data,column_names):
    # 组合特征
    features=data[column_names]
    rest_features=data.drop(column_names,axis=1)
    poly_transformer=PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
    poly_features=pd.DataFrame(poly_transformer.fit_transform(features),columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1,col,poly_features[col])
    return rest_features


def create_feature(df):
    new_train,new_test=df[:train_len],df[train_len:]
    return new_train,new_test



# 特征重要性
def plot_fea_importance(classifier,X_train):
    plt.figure(figsize=(10,12))
    name = "xgb"
    indices = np.argsort(classifier.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X_train.columns[indices][:40],
                    x=classifier.feature_importances_[indices][:40],orient='h')
    g.set_xlabel("Relative importance", fontsize=12)
    g.set_ylabel("Features", fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title(name + " feature importance")
    plt.show()


def evaluate_cv5_lgb(train_df, test_df, cols, test=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    y_test = 0
    oof_train = np.zeros((train_df.shape[0],))
    for i, (train_index, val_index) in enumerate(kf.split(train_df[cols])):
        X_train, y_train = train_df.loc[train_index, cols], train_df['label'].values[train_index]
        X_val, y_val = train_df.loc[val_index, cols], train_df['label'].values[val_index]
        xgb = XGBClassifier(learning_rate=0.12,
                            max_depth=6,
                            min_child_weight=3,
                            ubsample=0.98,
                            colsample_bytree=0.6)
        xgb.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                early_stopping_rounds=100, eval_metric=['auc'], verbose=True)
        y_pred = xgb.predict(X_val)

        if test:
            y_test += xgb.predict(test_df.loc[:, cols])
        oof_train[val_index] = y_pred

        if i==0:
            plot_fea_importance(xgb,X_train)
    print(train_df['label'].values)
    accuracy = accuracy_score(train_df['label'].values, oof_train.round())
    y_test /= 5
    print('5 Fold accuracy:', accuracy)
    return y_test


if __name__ == '__main__':
    train,test=create_feature(df_all)
    cols = [col for col in train.columns if col not in ['qid1','qid2','label','wid1','cid1','wid2','cid2']]
    y_test=evaluate_cv5_lgb(train,test,cols,True)
    test['label']=y_test
    test['label']=test['label'].apply(lambda x:1 if x>0.5 else 0)
    # test.rename(columns={'qid1':'question_id_1','qid2':'question_id_2'},inplace=True)
    test[['qid1','qid2','label']].to_csv('result/01_lgb_cv5.csv',columns=['qid1','qid2','label'], index=None)
