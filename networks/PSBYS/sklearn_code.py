# -*- codeing = utf-8 -*-
# @Time : 2024/9/6 10:48
# @Author : Luo_CW
# @File : sklearn_code.py
# @Software : PyCharm
# coding:utf-8

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def naviebayes():
    # 获取数据
    news = fetch_20newsgroups()
    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)
    # 对数据集进行特征抽取
    tf = TfidfVectorizer()
    # 以训练集当中词的列表进行每篇文章重要性统计['a','b','c','d']
    x_train = tf.fit_transform(x_train)
    print(tf.get_feature_names())
    x_test = tf.transform(x_test)
    # 进行朴素贝叶斯算法的预测
    mlt = MultinomialNB(alpha=1.0)
    print(x_train.toarray())
    mlt.fit(x_train, y_train)
    y_predict = mlt.predict(x_test)
    print("预测的文章类别为:", y_predict)
    # 得出准确率
    print("准确率为:", mlt.score(x_test, y_test))


if __name__ == '__main__':
    naviebayes()