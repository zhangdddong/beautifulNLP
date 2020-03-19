#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import numpy as np
import joblib

import global_parameters as config
from data_loader import get_sentences
from model_utils import doc_vector, sentence_vector, cos_dist, get_keywords, keyword_weight, len_weight
import data_utils


def get_first_summaries(text, stopwords, model):
    """
    根据相似度 关键词权重 以及 句子长度权重 先粗略计算出一个摘要句子列表
    :param text: 文档
    :param stopwords: 停用词
    :param model: 词向量模型
    :return: 摘要列表，按照权重倒排序[(句子, 权重), (句子, 权重), ...]
    """
    # [[index1, sentence1], [index2, sentence2], ... [-1, sentence]]
    sentences = get_sentences(text)

    # [sentence1, sentence2, ... ]
    sen_list = [x[1] for x in sentences]
    docvec = doc_vector(text, stopwords, model)
    sen_vec = []

    # 区分句首、句尾和句中的权重
    for i, sen in enumerate(sen_list):
        if i == 0:
            sen_vec.append(sentence_vector(sen, stopwords, model) * config.locFirst_weight)
        elif i == len(sen_list) - 1:
            sen_vec.append(sentence_vector(sen, stopwords, model) * config.locLast_weight)
        else:
            sen_vec.append(sentence_vector(sen, stopwords, model))

    # 文档向量和每句话的余弦相似度
    cos_list = [cos_dist(docvec, x) for x in sen_vec]

    keywords = get_keywords(text)
    # 关键字权重，每个句话包含多少文档关键字的衡量标准
    keyweights = [keyword_weight(x, keywords) for x in sen_list]
    # 长度权重，每句话和生成的摘要的长度的权重比值
    len_weights = [len_weight(x) for x in sen_list]

    final_weights = [cos * keyword * length for cos in cos_list for keyword in keyweights for length in len_weights]

    final_list = []
    for sen, weight in zip(sen_list, final_weights):
        final_list.append((sen, weight))

    final_list = sorted(final_list, key=lambda x: x[1], reverse=True)
    final_list = final_list[:config.first_num]

    return final_list


def MMR(final_list, stopwords, model):
    """
    定义MMR算法，保证句子的多样性。
    :param final_list: 初步摘要的句子列表 [(句子, 权重), (句子, 权重), ...]
    :param stopwords: 停用词表
    :param model: 词向量模型
    :return: 最终摘要句子列表
    """
    sen_list = [x[0] for x in final_list]
    weight_list = [x[1] for x in final_list]

    summary_list = []
    summary_list.append(sen_list[0])
    del sen_list[0]
    del weight_list[0]

    if config.last_num == 1:
        return summary_list
    else:
        for i in range(len(sen_list)):
            # [[sentence_vector1], [sentence_vector2], ... ]
            vec_list = [sentence_vector(x, stopwords, model) for x in sen_list]
            summary_vec = [sentence_vector(x, stopwords, model) for x in summary_list]
            scores = []

            for vec1 in vec_list:
                count = 0
                score = 0
                for vec2 in summary_vec:
                    score += config.alpha * weight_list[count] - \
                             (1 - config.alpha) * cos_dist(vec1, vec2)
                count += 1
                scores.append(score / len(summary_vec))

            scores = np.array(scores)
            index = np.argmax(scores)
            summary_list.append(sen_list[index])

            del sen_list[index]
            del weight_list[index]
        return summary_list[:config.last_num]


def get_last_summaries(text, final_list, stopwords, model):
    """
    获取最终的摘要
    :param text:
    :param final_list:
    :param stopwords: 停用词
    :param model: 词向量模型
    :return: list [summary1, summary2, summary3, ... ]
    """
    if config.use_MMR:
        results = MMR(final_list, stopwords, model)
    else:
        results = final_list[:config.last_num]
        results = [x[0] for x in results]

    sentences = get_sentences(text)
    summaries = []

    for summary in results:
        for sentence in sentences:
            if summary == sentence[1]:
                summaries.append((summary, sentence[0]))

    summaries = sorted(summaries, key=lambda x: x[1])
    summaries = [x[0] for x in summaries]

    return summaries


if __name__ == '__main__':

    stopwords = data_utils.read_stopwords()
    model = joblib.load(config.w2v_model_path)

    text = "记得很小的时候，我到楼下去玩，一不小心让碎玻璃割伤了腿，疼得我“哇哇”大哭。爸爸问讯赶来，把我背到了医院，仔仔细细地为我清理伤口《爸爸是医生》、缝合、包扎，妈妈则在一旁流眼泪，一副胆战心惊的样子。我的腿慢慢好了，爸爸妈妈的脸上，才渐渐有了笑容。 一天下午，放学时，忽然下起了倾盆大雨。我站在学校门口，喃喃自语：“我该怎么办？”正在我发愁的时候，爸爸打着伞来了。“儿子，走，回家！”我高兴得喜出望外。这时，爸爸又说话了：“今天的雨太大了，地上到处是水坑，我背你回家！”话音未落，爸爸背起我就走了。一会儿，又听到爸爸说：“把伞往后挪一点，要不挡住我眼了。”我说：“好！”回到家，发现爸爸的衣服全湿透了，接连打了好几个喷嚏。我的眼泪涌出来了。 “可怜天下父母心”，这几年里，妈妈为我洗了多少衣服，爸爸多少次陪我学习玩耍，我已经记不清了。让我看在眼里、记在心里的是妈妈的皱纹、爸爸两鬓的白发。我的每一步成长，都包含了父母太多的辛勤汗水和无限爱心，“可怜天下父母心”！没有人怀疑，父母的爱是最伟大的、最无私的！"

    final_list = get_first_summaries(text, stopwords, model)

    # 最终摘要
    print(get_last_summaries(text, final_list, stopwords, model))
