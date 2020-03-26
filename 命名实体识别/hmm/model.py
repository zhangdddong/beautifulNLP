#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import numpy as np

from model_utils import log


def viterbi(x, pi, A, B, word2id, id2tag, tag2id):
    """
    viterbi algorithm
    :param x: user input string / sentence
    :param pi: initial probability of tags
    :param A: 给定tag,每个单词出现的概率(发射概率)
    :param B: tag之间的转移概率(转移概率)
    :param word2id: 单词到索引的映射
    :param id2tag: 索引到标签的映射
    :param tag2id: 标签到索引的映射
    :return:
    """
    num_words = len(word2id)  # 训练集中词的数量
    num_tags = len(tag2id)  # 训练集中的标签的数量

    x = [word2id[word] for word in x]
    T = len(x)

    dp = np.zeros((T, num_tags))
    ptr = np.array([[0 for x in range(num_tags)] for y in range(T)])

    for j in range(num_tags):
        dp[0][j] = log(pi[j]) + log(A[j][x[0]])

    for i in range(1, T):
        for j in range(num_tags):
            dp[i][j] = -9999
            for k in range(num_tags):
                score = dp[i - 1][k] + log(B[k][j]) + log(A[j][x[i]])
                if score > dp[i][j]:
                    dp[i][j] = score
                    ptr[i][j] = k

    # decoding: 把最好的tag sequence打印出来
    best_seq = [0] * T  # best_seq = [1, 5, 2, 234,...]

    # step1: 找出对应于最后一个字的标签
    best_seq[T - 1] = np.argmax(dp[T - 1])

    # step2: 通过从后到前的循环依次求出每个字的标签
    for i in range(T - 2, -1, -1):
        best_seq[i] = ptr[i + 1][best_seq[i + 1]]

    # 到目前为止，best_seq存放了对应于x的标签序列
    for i in range(len(best_seq)):
        print(id2tag[best_seq[i]])
