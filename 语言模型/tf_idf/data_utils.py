#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'
import re


def get_config():
    """
    Loading config
    配置信息，如文件的存储路径
    :return: config information
    """
    config = dict()
    config['dilate_title'] = 2    # terms in the title is multiplied by X
    config['dilate_first'] = 2    # terms in the first paragraph is multiplied by X
    config['data'] = './data/data'  # raw data
    config['url'] = './res/url'    # url information
    config['title'] = './res/title'    # title information
    config['content'] = './res/content'    # content(title and article)
    config['content_clean'] = './res/content_clean'    # cleaned content
    config['content_filter'] = './res/content_filter'  # filtered content
    config['term_list'] = './res/term_list'    # term list
    config['content_stemming'] = './res/content_stemming'  # stemming content
    config['idf'] = './res/idf_file'
    config['tf_idf'] = './res/tf_idf_file'
    config['vector_normal'] = True
    return config


def data_clean(text):
    """
    Clean text
    清洗文本
    :param text: the string of text
    :return: text string after cleaning
    """
    # 包含网址舍去
    if re.search(r'pic\.twitter\.com', text):
        text = ''
    if re.match(r'^Source', text):
        text = ''
    if re.search(r'@', text):
        text = ''

    # acronym
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"can’t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"what’s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"What’s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"’ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"n’t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"i’m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"I’m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"’re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"’d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"’ll", " will ", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)

    # spelling correction
    text = re.sub(r"ph\.d", "phd", text)
    text = re.sub(r"PhD", "phd", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" fb ", " facebook ", text)
    text = re.sub(r"facebooks", " facebook ", text)
    text = re.sub(r"facebooking", " facebook ", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" us ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" U\.S\. ", " america ", text)
    text = re.sub(r" US ", " america ", text)
    text = re.sub(r" American ", " america ", text)
    text = re.sub(r" America ", " america ", text)
    text = re.sub(r" mbp ", " macbook-pro ", text)
    text = re.sub(r" mac ", " macbook ", text)
    text = re.sub(r"macbook pro", "macbook-pro", text)
    text = re.sub(r"macbook-pros", "macbook-pro", text)
    text = re.sub(r" 1 ", " one ", text)
    text = re.sub(r" 2 ", " two ", text)
    text = re.sub(r" 3 ", " three ", text)
    text = re.sub(r" 4 ", " four ", text)
    text = re.sub(r" 5 ", " five ", text)
    text = re.sub(r" 6 ", " six ", text)
    text = re.sub(r" 7 ", " seven ", text)
    text = re.sub(r" 8 ", " eight ", text)
    text = re.sub(r" 9 ", " nine ", text)
    text = re.sub(r"googling", " google ", text)
    text = re.sub(r"googled", " google ", text)
    text = re.sub(r"googleable", " google ", text)
    text = re.sub(r"googles", " google ", text)
    text = re.sub(r"dollars", " dollar ", text)
    text = re.sub(r"[0-9]", " ", text)

    # punctuation
    text = re.sub(r"€", " ", text)
    text = re.sub(r"%", " ", text)
    text = re.sub(r"‘", " ", text)
    text = re.sub(r"’", " ", text)
    text = re.sub(r"”", " ", text)
    text = re.sub(r"“", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"/", " ", text)
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"\^", " ", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r";", " ", text)
    text = re.sub(r"\(", " ", text)
    text = re.sub(r"\)", " ", text)
    text = re.sub(r"&", " & ", text)
    text = re.sub(r"\|", " | ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"\+", " + ", text)

    # symbol replacement
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\|", " or ", text)
    text = re.sub(r"=", " equal ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r'\s+', ' ', text)

    text = ' '.join(re.findall(r'[a-zA-Z\d]+', text))

    return text


def get_stop_word():
    """
    去除停用词，将停用词以集合的形式返回
    :return: stop words
    """
    stop_word = set()
    for line in open('./data/stop_word_list', encoding='UTF-8'):
        stop_word.add(line.strip())
    return stop_word
