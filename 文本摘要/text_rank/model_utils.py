#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import numpy as np
import jieba
from jieba import analyse
import joblib

import global_parameters as config
from data_loader import get_sentences
import data_utils


def sentence_vector(sentence, stop_words, model):
    """
    生成句向量方法
    根据词向量模型 和 给定句子 生成句向量
    :param sentence:
    :param stop_words: 停用词列表 之所以传入形式  是因为可以再程序启动后 只需要在外部加载一次
    :param model: 词向量模型，之所以通过传参数 是为了项目上线时候 可以再外部加载一次模型，不能在代码内部加载
    :return: 返回句子的向量  是 np.array() 格式的
    """
    vector = np.zeros(100, )    # 100和词向量维度相同

    if config.use_words_vector:
        if config.use_stopwords:
            content = jieba.lcut(sentence)
            count = 0
            for word in content:
                if word not in stop_words:
                    count += 1
                    vector += np.array(model[word])
            if count == 0:
                return np.zeros(100, )
            return vector / count   # 防止vector值过大
        else:
            content = jieba.lcut(sentence)
            length = len(content)
            for word in content:
                vector += np.array(model[word])
            return vector / length


def doc_vector(text, stop_words, model):
    """
    计算文档向量，句子向量求平均
    :param text: 需要计算的文档
    :param stop_words: 停用词表
    :param model: 词向量模型
    :return: 文档向量
    """
    sen_list = get_sentences(text)
    sen_list = [x[1] for x in sen_list]
    vector = np.zeros(100, )
    length = len(sen_list)
    for sentence in sen_list:
        sen_vec = sentence_vector(sentence, stop_words, model)
        vector += sen_vec
    return vector / length


def cos_dist(vec1, vec2):
    """
    定义余弦函数  dis(A, B) = A * B / |A| * |B|
    :param vec1: vector1
    :param vec2: vector2
    :return: 两个向量的余弦值
    """
    dist1 = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) + np.linalg.norm(vec2)))
    return dist1


def get_keywords(text):
    """
    返回关键字信息
    :param text: 需要提取关键字的文档，str类型
    :return:
    """
    if config.keyword_type == 0:
        keywords = analyse.textrank(text)
        return keywords
    else:
        print('tf_idf方法没完成')


def keyword_weight(sentence, keywords):
    """
    获取一个句子在一篇文档里的关键字权重值
    :param sentence: str type 一句话
    :param keywords: list type [keyword1, keyword2, ... ] 一篇文章的全部关键词
    :return:
    """
    n = 0
    for keyword in keywords:
        n += sentence.count(keyword)

    return n / len(keywords)


def len_weight(sentence):
    """
    计算句子长度，定义句子长度权重
    :param sentence: 求解的句子，一句话
    :return: 句子长度权重
    """
    # 求解的句子长度小于摘要句子长度
    if len(sentence) <= config.summary_len:
        if len(sentence) / config.summary_len > config.minLeng_weight:
            return len(sentence) / config.summary_len
        else:
            return config.minLeng_weight
    else:
        """
        句子长度大于摘要长度
        如果句子长度大于我们想要的摘要句子长度的两倍，那他的权重就是负数，基本不会被选为摘要
        return 1 - (len(sentence) - global_parameters.summary_len) / global_parameters.summary_len
        采用下面策略，起码保证句子长度权重值不会为负数，最低为0.5
        """
        if 1 - (len(sentence) - config.summary_len) / \
                config.summary_len > config.minLeng_weight:
            return 1 - (len(sentence) - config.summary_len) / config.summary_len
        else:
            return config.minLeng_weight


if __name__ == '__main__':
    stop_words = data_utils.read_stopwords()

    model = joblib.load(config.w2v_model_path)
    text = "记得很小的时候，我到楼下去玩，一不小心让碎玻璃割伤了腿，疼得我“哇哇”大哭。爸爸问讯赶来，把我背到了医院，仔仔细细地为我清理伤口《爸爸是医生》、缝合、包扎，妈妈则在一旁流眼泪，一副胆战心惊的样子。我的腿慢慢好了，爸爸妈妈的脸上，才渐渐有了笑容。 一天下午，放学时，忽然下起了倾盆大雨。我站在学校门口，喃喃自语：“我该怎么办？”正在我发愁的时候，爸爸打着伞来了。“儿子，走，回家！”我高兴得喜出望外。这时，爸爸又说话了：“今天的雨太大了，地上到处是水坑，我背你回家！”话音未落，爸爸背起我就走了。一会儿，又听到爸爸说：“把伞往后挪一点，要不挡住我眼了。”我说：“好！”回到家，发现爸爸的衣服全湿透了，接连打了好几个喷嚏。我的眼泪涌出来了。 “可怜天下父母心”，这几年里，妈妈为我洗了多少衣服，爸爸多少次陪我学习玩耍，我已经记不清了。让我看在眼里、记在心里的是妈妈的皱纹、爸爸两鬓的白发。我的每一步成长，都包含了父母太多的辛勤汗水和无限爱心，“可怜天下父母心”！没有人怀疑，父母的爱是最伟大的、最无私的！"

    print(doc_vector(text, stop_words, model))