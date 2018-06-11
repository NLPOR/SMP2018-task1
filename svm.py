"""
@author = 'XXY'
@contact = '529379497@qq.com'
@researchFie1d = 'NLP DL ML'
@date= '2017/12/21 10:18'
"""
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import json
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
# from predictor import data
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import pickle
from data_helper import get_data
from gensim.models.word2vec import Word2Vec
import thulac
from sklearn.model_selection import train_test_split

dim = 5000
def cut_text(alltext):
    # 分词
    count = 0
    cut = thulac.thulac(seg_only=True)
    train_text = []
    for text in alltext:
        count += 1
        if count % 2000 == 0:
            print(count)
        train_text.append(cut.cut(text, text=True))
    return train_text


def train_tfidf(train_data):
    tfidf = TFIDF(
        min_df=5,
        max_features=dim,
        ngram_range=(1, 2),
        use_idf=1,
        smooth_idf=1
    )
    tfidf.fit(train_data)
    return tfidf


def train_word2vec(train_data):
    model = Word2Vec(train_data, size=128, window=5, min_count=5, workers=4)
    return model


def train_SVC(vec, label):
    SVC = LinearSVC(C=100)
    SVC.fit(vec, label)
    return SVC


def get_word2vec(content):
    word2vec = Word2Vec.load('predictor/model/wiki.zh.seg_200d.model')
    res = np.zeros([200])
    count = 0
    # word_list = content.split()
    for word in content:
        if word in word2vec:
            res += word2vec[word]
            count += 1
    return pd.Series(res / count)


if __name__ == '__main__':
    print('reading...')
    all_text, y, label, label_to_int, int_tolabel = get_data(file='./data/train_data.csv')
    print('cut text...')
    train_data = cut_text(all_text)
    # train_data = [line.split() for line in train_data]
    print('get tfidf...')
    tfidf = train_tfidf(train_data)
    print('saving tfidf model')
    joblib.dump(tfidf, 'model/tfidf_5000.model')

    X_train, X_dev, y_train, y_dev = train_test_split(train_data, y, random_state=10, test_size=0.1)
    train_vec = tfidf.transform(X_train)
    test_vec = tfidf.transform(X_dev)
    print('training SVC')
    svm = train_SVC(train_vec, y_train)
    y_pre = svm.predict(test_vec)
    print(classification_report(y_dev, y_pre))
    print("saving svm model")
    joblib.dump(svm, 'model/svm_5000.model')
