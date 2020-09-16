#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-09-16 8:22
# @Description: In User Settings Edit
# @Software : PyCharm
import re
import math
import jieba


text = """
自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。
它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，
所以它与语言学的研究有着密切的联系，但又有重要的区别。
自然语言处理并不是一般地研究自然语言，
而在于研制能有效地实现自然语言通信的计算机系统，
特别是其中的软件系统。因而它是计算机科学的一部分。
"""


def get_sentences(doc):
    line_break = re.compile('[\r\n]')
    delimiter = re.compile('[，。？！；]')
    sentences = []
    for line in line_break.split(doc):
        line = line.strip()
        if not line:
            continue
        for sent in delimiter.split(line):
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)
    return sentences


# def filter_stop(words):
#     return list(filter(lambda x: x not in stop, words))


class IFIDF(object):
    def __init__(self, docs):
        self.D = len(docs)
        self.docs = docs
        self.f = []    # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {}   # 存储每个词及出现了该词的文档数量
        self.idf = {}  # 存储每个词的idf值
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D) - math.log(v + 1)

    def get_tfidf(self):
        for index, sent in enumerate(self.docs):
            vec = [self.f[index][word] * self.idf[word] for word in sent]
            print(vec)


if __name__ == '__main__':
    sents = get_sentences(text)
    doc = []
    for sent in sents:
        words = list(jieba.cut(sent))
        # words = filter_stop(words)
        doc.append(words)
    m = IFIDF(doc)
    m.get_tfidf()
