#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'


def get_features(file_path):
    """
    :param file_path: 文件路径
    :return:
    """
    word2id, id2word = {}, {}
    tag2id, id2tag = {}, {}

    for line in open(file_path, encoding='UTF-8'):
        items = line.split(' ')
        word, tag = items[0], items[1].rstrip()
        if word not in word2id:
            word2id[word] = len(word2id)
            id2word[len(id2word)] = word
        if tag not in tag2id:
            tag2id[tag] = len(tag2id)
            id2tag[len(id2tag)] = tag

    return word2id, id2word, tag2id, id2tag
