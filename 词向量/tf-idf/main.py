#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'
import os
from data_loader import data_load
from data_loader import data_clean_content
from data_loader import filter_stop_word
from data_loader import create_term_list
from data_utils import get_config
from porter_stemmer import stemming
from model import get_content
from model import get_tf
from model import get_idf
from model import create_idf
from model import create_tf_idf
import time


def run():
    """
    如果文件不存在，则创建
    :return:
    """
    if not os.path.exists('./res'):
        os.makedirs('res')
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

    documents = get_content(config)
    tf_documents = get_tf(documents)
    if not os.path.exists(config['idf']):
        create_idf(config, documents)
    idf_documents = get_idf(config)
    if not os.path.exists(config['tf_idf']):
        create_tf_idf(config, tf_documents, idf_documents, documents)


if __name__ == '__main__':
    run()
