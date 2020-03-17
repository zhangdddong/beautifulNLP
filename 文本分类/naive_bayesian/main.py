#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import jieba.posseg as pesg
import nltk
from nltk import data

data.path.append(r'E:\dataset\03_nltk')


def read_data(filename):
    """
    读取数据
    :param filename:
    :return: list [(text1: str, class1: str), (text2, class2), ... ]
    """
    data = []
    with open(filename, encoding='UTF-8') as f:
        for line in f:
            label, sentences = line.split('\t')
            sentence_list = sentences.split('；')
            data.extend([[sentence, label] for sentence in sentence_list if sentence])
    return data


def delte_stop_word(sentence):
    """
    处理停用词表
    :param sentence:
    :return:
    """
    with open('./data/baidu_stopwords.txt', encoding='UTF-8') as f:
        stop_word = [line.strip() for line in f if line]
    for word in stop_word:
        sentence.replace(word, '')
    return sentence


def get_word_features(sentence):
    """
    特征选择，这里使用分词后的词向作为特征。
    :param sentence:
    :return:
    """
    data = {}
    sentence = delte_stop_word(sentence)
    sen_list = pesg.cut(sentence)
    for word, tag in sen_list:
        data[tag] = word
    return data


def get_features_sets(filename):
    """
    构建训练数据
    :param filename:
    :return: list [[{tag1: word1, tag2: word2, ... }, class1], [{...}, class2], ... ]
    """
    feature_sets = []
    for sentence, label in read_data(filename):
        feature = get_word_features(sentence)
        feature_sets.append([feature, label])
    return feature_sets


if __name__ == '__main__':
    classifier = nltk.NaiveBayesClassifier.train(get_features_sets('./data/data.txt'))

    predict_label = classifier.classify(get_word_features('请问明天的天气怎么样？'))
    print(predict_label)
    print(classifier.prob_classify(get_word_features('请问明天的天气怎么样？')).prob(predict_label))

    while True:
        print('请输入您预测的句子: ')
        sentence = input()

        predict_label = classifier.classify(get_word_features(sentence))
        prob = classifier.prob_classify(get_word_features(sentence)).prob(predict_label)

        print('文本 <%s> 预测类别为: %s 的概率为 %f' % (sentence, predict_label, prob))
