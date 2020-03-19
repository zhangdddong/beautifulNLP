#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'


# 停用词路径
stop_words_path = './data/stop_words.txt'

# 原始文章存放路径
origin_data_path = 'origin_data/origin_data'

# 处理好的用于训练词向量的数据存放路径，格式是每行一篇文章
text_path = 'data/text'

# 词向量模型存放路径
w2v_model_path = 'model/w2v.model'

"""
选择训练字向量还是词向量，之所以选择是否训练字向量，是为了防止OOV问题
训练字向量的话可以避免该问题，True表示训练的词向量，False表示训练字向量
此处注意：如果训练字向量，那就不需要去除停用词
"""
use_words_vector = True

# 是否去掉停用词，默认是True，表示去掉停用词
use_stopwords = True

# 生成文档向量时，断句用的符号，这个要根据给定文章的格式进行调整，下面是中英文版的
break_points = [',', '.', '!', '?', ';', '，', '。', '！', '？', '；']

# 提取关键字的方法是textRank和tf-idf，0表示textRank 1表示tf-idf，本代码只是暂时实现了textRank方法，如果需要可以在utils中自己实现
keyword_type = 0

# 设置位置权重
locFirst_weight = 1.2
locLast_weight = 1.2

# 设定我们想要的摘要句子长度，根据该长度可以求解出长度的权重
summary_len = 10

# 每条句子由于长度产生的最小权重值
minLeng_weight = 0.5

# 第一次摘要列表句子个数
first_num = 10

# 最终我们得到的摘要句子个数
last_num = 5

# 是否保持句子多样性，采用MMR，默认采用该技术
use_MMR = True

# 定义MMR算法的alpha值，alpha值越小，说明需要多样性的程度越大
alpha = 0.5