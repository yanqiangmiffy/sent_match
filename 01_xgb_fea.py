# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 01_xgb_fea.py
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
import difflib # python官方库difflib的类SequenceMatcher  比较文本的距离
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.corpus import stopwords
from simhash import Simhash

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
    计算相似度:doc2vec cos eu
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
    qratios=[]
    wratios=[]
    partial_ratios=[]
    partial_token_set_ratios=[]
    partial_token_sort_ratios=[]
    token_set_ratios=[]
    token_sort_ratios=[]
    for wid1,wid2 in tqdm(zip(df_all['wid1'],df_all['wid2'])):
        ratios.append(fuzz.ratio(wid1,wid2))
        qratios.append(fuzz.QRatio(wid1,wid2))
        wratios.append(fuzz.WRatio(wid1,wid2))
        partial_ratios.append(fuzz.partial_ratio(wid1,wid2))
        partial_token_set_ratios.append(fuzz.partial_token_set_ratio(wid1,wid2))
        partial_token_sort_ratios.append(fuzz.partial_token_sort_ratio(wid1,wid2))
        token_set_ratios.append(fuzz.token_set_ratio(wid1,wid2))
        token_sort_ratios.append(fuzz.token_sort_ratio(wid1,wid2))
    return ratios,qratios,wratios,partial_ratios,partial_token_set_ratios,\
           partial_token_sort_ratios,token_set_ratios,token_sort_ratios


def get_diffs():
    """获取diff 距离"""
    print("get diffs ratios...")
    seq=difflib.SequenceMatcher()
    diffs=[]
    for wid1,wid2 in tqdm(zip(df_all['wid1'],df_all['wid2'])):
        seq.set_seqs(wid1.lower(),wid2.lower())
        diffs.append(seq.ratio())
    return diffs


def get_len(df_all):
    """
    文本长度
    :return:
    """
    print("get_len...")

    # 长度特征
    df_all['word_len1'] = df_all.wid1.map(lambda x: len(str(x)))  # word长度
    df_all['word_len2'] = df_all.wid2.map(lambda x: len(str(x)))

    df_all['char_len1'] = df_all.cid1.map(lambda x: len(str(x)))  # 字符长度
    df_all['char_len2'] = df_all.cid2.map(lambda x: len(str(x)))

    # 差值特征
    df_all['word_len_diff_ratio'] = df_all.apply(
        lambda row: abs(row.word_len1 - row.word_len2) / (row.word_len1 + row.word_len2), axis=1)
    df_all['char_len_diff_ratio'] = df_all.apply(
        lambda row: abs(row.char_len1 - row.char_len2) / (row.char_len1 + row.char_len2), axis=1)
    return df_all

def gen_tfidf(df_all):
    """
    tfidf 特征
    :return:
    """
    print("gen tfidf...")

    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))

    questions_txt = pd.Series(
        df_all['wid1'].tolist() +
        df_all['wid2'].tolist()
    ).astype(str)

    tfidf.fit_transform(questions_txt)

    tfidf_sum1 = []
    tfidf_sum2 = []
    tfidf_mean1 = []
    tfidf_mean2 = []
    tfidf_len1 = []
    tfidf_len2 = []

    for index, row in df_all.iterrows():
        tfidf_q1 = tfidf.transform([str(row['wid1'])]).data
        tfidf_q2 = tfidf.transform([str(row['wid2'])]).data

        tfidf_sum1.append(np.sum(tfidf_q1))
        tfidf_sum2.append(np.sum(tfidf_q2))
        tfidf_mean1.append(np.mean(tfidf_q1))
        tfidf_mean2.append(np.mean(tfidf_q2))
        tfidf_len1.append(len(tfidf_q1))
        tfidf_len2.append(len(tfidf_q2))

    df_all['tfidf_sum1'] = tfidf_sum1
    df_all['tfidf_sum2'] = tfidf_sum2
    df_all['tfidf_mean1'] = tfidf_mean1
    df_all['tfidf_mean2'] = tfidf_mean2
    df_all['tfidf_len1'] = tfidf_len1
    df_all['tfidf_len2'] = tfidf_len2

    return df_all

# ------>simhash
def tokenize(sequence):
    words = word_tokenize(sequence)
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return filtered_words

def clean_sequence(sequence):
    tokens = tokenize(sequence)
    return ' '.join(tokens)

def get_word_ngrams(sequence, n=3):
    tokens = tokenize(sequence)
    return [' '.join(ngram) for ngram in ngrams(tokens, n)]

def get_character_ngrams(sequence, n=3):
    sequence = clean_sequence(sequence)
    return [sequence[i:i+n] for i in range(len(sequence)-n+1)]


def caluclate_simhash_distance(sequence1, sequence2):
    return Simhash(sequence1).distance(Simhash(sequence2))

def get_word_distance(questions):
    # 词距离
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return caluclate_simhash_distance(q1, q2)

def get_word_2gram_distance(questions):
    # word_ngrams距离
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)

def get_char_2gram_distance(questions):
    # char_2gram距离
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)

def get_word_3gram_distance(questions):
    # word_3gram距离
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)

def get_char_3gram_distance(questions):
    # char_3gram距离
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)

# ------>simhash




doc_sims,cos_sim,eu_dis=calculate_sim()
ratios,qratios,wratios,partial_ratios,partial_token_set_ratios,\
           partial_token_sort_ratios,token_set_ratios,token_sort_ratios=get_fuzzy_ratios()
diffs=get_diffs()

df_all['doc_sims']=doc_sims
df_all['cos_sim']=cos_sim
df_all['eu_dis']=eu_dis
df_all['ratios'],df_all['qratios'],df_all['wratios'],df_all['partial_ratios']=ratios,qratios,wratios,partial_ratios
df_all['pt_set_r'],df_all['pt_sort_r'],df_all['t_set_r'],df_all['t_sort_r']=partial_token_set_ratios,partial_token_sort_ratios,\
                                                                            token_set_ratios,token_sort_ratios
df_all['diffs']=diffs

df_all=get_len(df_all) # 获取文本长度特征
df_all=gen_tfidf(df_all)

df_all['words'] = df_all['wid1'] + '_split_tag_' + df_all['wid2']
df_all['simhash_tokenize_distance'] = df_all['words'].apply(get_word_distance)
df_all['simhash_word_2gram_distance'] = df_all['words'].apply(get_word_2gram_distance)
df_all['simhash_char_2gram_distance'] = df_all['words'].apply(get_char_2gram_distance)
df_all['simhash_word_3gram_distance'] = df_all['words'].apply(get_word_3gram_distance)
df_all['simhash_char_3gram_distance'] =df_all['words'].apply(get_char_3gram_distance)

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
    cols = [col for col in df_all.columns if col not in ['qid1','qid2','label','wid1','cid1','wid2','cid2','words']]
    df_all[cols+['qid1','qid2','label']].to_csv('feature.csv',index=None)
    test['label']=evaluate_cv5_lgb(train,test,cols,True)
    test['label']=test['label'].apply(lambda x:1 if x>0.5 else 0)
    # test.rename(columns={'qid1':'question_id_1','qid2':'question_id_2'},inplace=True)
    test[['qid1','qid2','label']].to_csv('result/01_lgb_cv5.csv',columns=['qid1','qid2','label'], index=None)
