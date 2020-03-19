#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import os
import jieba
import joblib
from gensim.models import Word2Vec
import global_parameters as config


def read_stopwords():
    """
    读取停用词列表
    :return: 停用词列表 [word1, word2, ... ]
    """
    with open(config.stop_words_path, 'r', encoding='UTF-8') as f:
        stop_words = [line.strip() for line in f if line]
    return stop_words


def get_sentences():
    """
    读取原始数据，并将数据处理成word2vec模型需要的格式
    :return: [batch_size, sentence_length]
    """
    # 判断是否需要去掉停用词
    stop_words = []
    if config['use_stopwords:']:
        stop_words = read_stopwords()

    sentences = []

    # 分两种情况：词向量和字向量
    if config['use_words_vector']:
        with open(config['text_path'], 'r', encoding='UTF-8') as f:
            for line in f:
                if line:
                    content = line.strip()
                    content = jieba.lcut(content)
                    # 去掉停用词
                    if config['use_stopwords']:
                        for word in content:
                            if word in stop_words:
                                content.remove(word)
                    # 处理好的content加入sentences
                    if content:
                        sentences.append(content)
    else:
        with open(config['text_path'], 'r', encoding='UTF-8') as f:
            for line in f:
                if line:
                    content = line.strip()
                    # 字向量直接将content中的内容加进去
                    content = [x for x in content]
                    if content:
                        sentences.append(content)
    return sentences


def train():
    """
    训练好词向量模型 并 保存到指定位置
    :return:
    """
    sentences = get_sentences()

    # 训练模型
    model = Word2Vec(sentences, size=100, window=3, min_count=1)

    # 保存模型
    joblib.dump(model, config['w2v_model_path'])


if __name__ == '__main__':

    # 如果模型不存在，就开始训练
    if not os.path.exists(config['w2v_model_path']):
        train()

    # 加载模型
    model = joblib.load(config['w2v_model_path'])

    # 查看一下词向量
    print(model['记得'])
