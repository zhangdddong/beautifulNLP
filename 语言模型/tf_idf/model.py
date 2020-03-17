#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'
import math
import json
import os
from data_utils import get_config
import numpy as np


def get_content(config):
    """
    获取文章的内容 将标题*X  第一段*Y   将内容存储到list中  list是一个二维矩阵 形状[num_documents, document_words]
    num_documents： 表示文档数量
    document_words： 为每个文档中单词的数量
    :param config:
    :return: documents [num_documents, document_words]
    """
    dilate = 2
    X = config['dilate_title']
    Y = config['dilate_first']
    documents = []
    document = []
    for line in open(config['content_stemming'], encoding='UTF-8'):
        if line == '\n':
            dilate = 2
            documents.append(document)
            document = []
        else:
            if dilate == 2:
                line = line.strip().split()
                document.extend(line * X)
                dilate -= 1
            elif dilate == 1:
                line = line.strip().split()
                document.extend(line * Y)
                dilate -= 1
            else:
                document.extend(line.strip().split())
    if document:
        documents.append(document)
    return documents


def get_tf(documents):
    """
    计算tfidf的值 并将值存储到3维矩阵中，矩阵的形状[num_documents, document_words, 2]
    num_documents: 文档的数量
    document_words： 每个文章中的单词数量
    2: 表示  单词-tf值
    :param documents:
    :return: [num_documents, document_words, 2]
    """
    tf_documents = []
    for document in documents:
        total = len(document)
        tf_document = []
        s = set()
        for word in document:
            if word not in s:
                s.add(word)
                tf_document.append([word, float(document.count(word)) / total])
        tf_documents.append(tf_document)
    return tf_documents


def create_idf(config, documents):
    """
    create idf file
    统计每个字的idf值，并存入文件
    通过字典的json格式存储 读取时也会时字典格式 方便
    :param config:
    :param documents:
    :return:
    """
    total = len(documents)
    idf = dict()
    words = []
    for document in documents:
        words.extend(document)
    s = set(words)
    for word in s:
        word_document = 0
        for i in range(total):
            if word in documents[i]:
                word_document += 1
        idf[word] = math.log10(float(total) / word_document)
    with open(config['idf'], 'w', encoding='UTF-8') as f:
        json.dump(idf, f, ensure_ascii=False, indent=4)


def get_idf(config):
    """
    get idf
    从文件中获取idf的值，json格式读取 可以直接是字典格式
    :param config:
    :return: idf dict
    """
    with open(config['idf'], encoding='UTF-8') as f:
        return json.load(f)


def create_tf_idf(config, tf_documents, idf_documents, documents):
    """
    计算文档的tf-idf的值：
    1) 从term_list中读取词汇表
    2) 每个文档中分别找到对应的单词，并计算if-idf，文档中不存在的填充0
    :param config:
    :param tf_documents:
    :param idf_documents:
    :param documents:
    :return:
    """
    tf_idf_documents = []
    term_list = []
    for line in open(config['term_list'], encoding='UTF-8'):
        term_list.append(line.strip())
    for i, document in enumerate(documents):
        tf_idf_document = []
        for word in term_list:
            if word in document:
                tf = [tf for tf in tf_documents[i] if tf[0] == word][0][1]
                idf = idf_documents[word]
                tf_idf_document.append(tf * idf)    # 文档中存在的计算tf-idf
            else:
                tf_idf_document.append(0)   # 文档中不存在的填充0
        tf_idf_documents.append(tf_idf_document)

    # vector normalization (optional) and config['vector_normal'] is bool type
    # 如果config['vector_normal']为True，则进行vector normalization
    if config['vector_normal']:
        for i, tf_idf_document in enumerate(tf_idf_documents):
            tf_idf_np = np.array(tf_idf_document)
            tf_idf_np = tf_idf_np / np.linalg.norm(tf_idf_np, ord=2)
            tf_idf_documents[i] = tf_idf_np.tolist()
    # 将if-idf的值 存入文件 每个文档对应一行
    with open(config['tf_idf'], 'w', encoding='UTF-8') as f:
        for tf_idf_document in tf_idf_documents:
            line = ' '.join([str(tf_idf) for tf_idf in tf_idf_document])
            f.write(line + '\n')


if __name__ == '__main__':
    config = get_config()
    documents = get_content(config)
    tf_documents = get_tf(documents)
    if not os.path.exists(config['idf']):
        create_idf(config, documents)
    idf_documents = get_idf(config)
    if not os.path.exists(config['tf_idf']):
        create_tf_idf(config, tf_documents, idf_documents, documents)
