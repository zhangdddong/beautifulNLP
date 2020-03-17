#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'
import os
from data_utils import data_clean
from data_utils import get_config
from data_utils import get_stop_word
from create_term_list import create_term_list
from porter_stemmer import stemming


def data_load(config):
    """
    Read data from file, and make 3 files: url, title, content
    加载数据，并将链接  标题 内容分开 分别写入文件中
    :param config: config information
    :return:
    """
    url = open(config['url'], 'w', encoding='UTF-8')
    title = open(config['title'], 'w', encoding='UTF-8')
    content = open(config['content'], 'w', encoding='UTF-8')

    with open(config['data'], encoding='UTF-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line[:4] == 'http':
                url.write(line)
                title.write(lines[i + 1])
                if i > 0:
                    content.write('\n')
            else:
                content.write(line)

    url.close()
    title.close()
    content.close()


def data_clean_content(config):
    """
    Clean content and store
    将content内容清洗，并存入文件，调用data_utils中的data_clean方法
    :param config: config information
    :return:
    """
    if not os.path.isfile(config['content']):
        raise Exception('No File Exception')
    with open(config['content_clean'], 'w', encoding='UTF-8') as f:
        for line in open(config['content'], encoding='UTF-8'):
            if line == '\n':
                f.write('\n')
            else:
                line = line.strip()
                line = data_clean(line)
                if line:
                    f.write(line + '\n')


def filter_stop_word(config):
    """
    Filter stop word and store
    将清洗后的文件过滤 停用词表
    :param config: config information
    :return:
    """
    stop_word = get_stop_word()
    with open(config['content_filter'], 'w', encoding='UTF-8') as f:
        for line in open(config['content_clean'], encoding='UTF-8'):
            if line == '\n':
                f.write('\n')
            else:
                line = line.strip().split()
                line = [word for word in line if word not in stop_word]
                line = ' '.join(line)
                if line:
                    f.write(line + '\n')


if __name__ == '__main__':
    config = get_config()
    if not os.path.exists(config['url']) or not os.path.exists(
            config['title'] or not os.path.exists(config['content'])):
        data_load(config)
    if not os.path.exists(config['content_clean']):
        data_clean_content(config)
    if not os.path.exists(config['content_filter']):
        filter_stop_word(config)
    if not os.path.exists(config['content_stemming']):
        stemming(config)
    if not os.path.exists(config['term_list']):
        create_term_list(config)
