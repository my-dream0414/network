# -*- codeing = utf-8 -*-
# @Time : 2024/9/6 11:04
# @Author : Luo_CW
# @File : cn_weibo_test.py
# @Software : PyCharm

# 导包
import numpy as np
import pandas as pd
import re
import jieba
# 训练模型，朴素贝叶斯算法
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

# 停去词
stopword_path = "cn_stopwords.txt"

def load_stopwords(file_path):
    stop_words = []
    with open(file_path, encoding='UTF-8') as words:
        stop_words.extend([i.strip() for i in words.readlines()])
    return stop_words


# 去掉{% %}之间的内容
def clearContentWithSpecialCharacter(content):
    content = content.replace("{%", "1")
    content = content.replace("%}", "1")
    pattern = re.compile(r'(1)(.*)(1)')
    return pattern.sub(r'', content)


# 预处理
def review_to_text(review):
    stop_words = load_stopwords(stopword_path)

    # 去掉@用户名
    review = re.sub("(@[a-zA-Z0-9_\u4E00-\u9FA5]+)", '', review)

    # 去除英文
    review = re.sub("[^\u4e00-\u9fa5^a-z^A-Z]", '', review)

    # 分词
    review_cut = jieba.cut(review)

    # 去掉停用词
    if stop_words:
        all_stop_words = set(stop_words)
        words = [w for w in review_cut if w not in all_stop_words]
    return words

# 加载语料
def load_curpus(path):
    data = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            [_,sentiment,comment] = line.split(",",2)
            data.append((comment,int(sentiment)))
    return data

# 处理语料
def pro_curpus(path):
    data = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            [_,sentiment,comment] = line.split(",",2) #以，为分界  分成3份
            content = review_to_text(comment)
            data.append((content,int(sentiment)))
    return data


train_data = load_curpus("train.txt")
test_data = load_curpus("test.txt")

train_data_pro = pro_curpus("train.txt")
test_data_pro = pro_curpus("test.txt")



train_df = pd.DataFrame(train_data_pro,columns=['content','sentiment'])
test_df = pd.DataFrame(test_data_pro,columns=['content','sentiment'])

# 将文档处理成 tokens
data_str = [' '.join(content) for content,sentiment in train_data_pro ] + \
        [' '.join(content) for content,sentiment in test_data_pro ] # 以' '将切割的词连接(join)起来

# 词向量（数据处理）
vect = CountVectorizer(max_df = 0.8,min_df = 5)

# 训练模型，朴素贝叶斯算法
nb = BernoulliNB()

vect.fit_transform(data_str) # 对训练集用fit_transform 测试集用transform，否则词向量的维度会不等


# 生成训练集 测试集的 词向量
X_data, y_data = [], []
for content, sentiment in train_data_pro:
    X_data.append(" ".join(content))
    y_data.append(sentiment)
X_train = vect.transform(X_data)
y_train = y_data

X_data2,y_data2 =[],[]
for content,sentiment in test_data_pro:
    X_data2.append(" ".join(content))
    y_data2.append(sentiment)
X_test = vect.transform(X_data2)
y_test = y_data2

# 使用贝叶斯进行训练
nb.fit(X_train,y_train)
train_score = nb.score(X_train,y_train)
print(train_score)

# 打印 bad_cases
bad_cases = []
for i in range(X_train.shape[0]):
    if(nb.predict(X_train[i])!=y_train[i]):
#         print("[%s],[%s],[真实：%s]，[判断：%s]" %(review_data[i],X_train[i],y_train[i],nb.predict(X_train_vect[i])))
        print("[%s],[%s],[真实：%s]，[判断：%s]" %(train_data[i],X_data[i],y_train[i],nb.predict(X_train[i])))


# 测试模型
test_score = nb.score(X_test,y_test)
print(test_score)

from sklearn import metrics
result = nb.predict(X_test)
print(metrics.classification_report(y_test, result))

bad_cases = []
for i in range(result.shape[0]):
    if(result[i]!=y_test[i]):
        print("[%s],[真实：%s]，[判断：%s]" %(train_df['content'][i],y_test[i],result[i]))