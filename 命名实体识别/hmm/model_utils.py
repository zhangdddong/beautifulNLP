#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import numpy as np


def log(v):
    if v == 0:
        return np.log(v + 0.00000001)
    return np.log(v)


def get_matrix(word2id, tag2id, id2tag, file_path):
    """
    :param word2id:
    :param tag2id:
    :param id2tag:
    :param file_path:
    :return:
    """
    num_words = len(word2id)  # 训练集中词的数量
    num_tags = len(tag2id)  # 训练集中的标签的数量

    pi = np.zeros(num_tags)  # 初始状态概率矩阵
    A = np.zeros((num_tags, num_words))  # 发射概率矩阵
    B = np.zeros((num_tags, num_tags))  # 状态转移概率矩阵

    # 计算矩阵中的对应数据出现的次数
    prev_tag = ''
    for line in open(file_path, encoding='UTF-8'):
        items = line.split(' ')
        wordId, tagId = word2id[items[0]], tag2id[items[1].rstrip()]
        if prev_tag == '':
            pi[tagId] += 1
            A[tagId][wordId] += 1
        else:
            A[tagId][wordId] += 1
            B[tag2id[prev_tag]][tagId] += 1
        if items[0] == '.':
            prev_tag = ''
        else:
            prev_tag = id2tag[tagId]

    # normalize 将统计的个数化成概率
    pi = pi / sum(pi)
    for i in range(num_tags):
        A[i] /= sum(A[i])
        B[i] /= sum(B[i])
    # 到此为止，计算完了模型的所有参数：pi A B
    return pi, A, B
