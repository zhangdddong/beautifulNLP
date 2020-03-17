#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'
import re


def create_term_list(config):
    """
    Create term list
    从已经stemming的文件中抽取出term_list
    :param config:
    :return:
    """
    terms = set()   # 定义一个集合，防止此表重复
    for line in open(config['content_stemming'], encoding='UTF-8'):
        line = line.strip()
        if line:
            for word in line.split():
                terms.add(word)
    term_list = list(terms)     # 转换成list，变成有序
    term_list = sorted(term_list)   # 排序

    # 将词写入文件
    with open(config['term_list'], 'w', encoding='UTF-8') as f:
        for word in term_list:
            f.write(word.strip() + '\n')
