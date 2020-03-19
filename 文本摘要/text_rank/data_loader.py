#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import global_parameters as config


def get_sentences(text):
    """
    将文档切割成句子
    :param text: 需要断句的文档
    :return: list [[index1, sentence1], [index2, sentence2], ..., [-1, sentenceN]]
    """
    break_points = config.break_points

    # 将其标点换算成统一符号，便于分割
    for point in break_points:
        text = text.replace(point, '<POINT>')

    # 根据<POINT>断句
    sen_list = text.split('<POINT>')

    # 去掉断句后的空字符
    sen_list = [x for x in sen_list if x != '']

    res = []
    for i, word in enumerate(sen_list):
        if i != len(sen_list) - 1:
            res.append([i + 1, sen_list[i]])
        else:
            # 最后一句话 位置是-1
            res.append([-1, sen_list[i]])

    return res
