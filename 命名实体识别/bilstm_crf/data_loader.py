#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import data_utils


def load_sentences(path):
    """
    加载数据集合，每一行至少包含一个字符和一个标记
    句子和句子之间用空格分隔
    最后返回句子的集合
    :param path:
    :return:
    """
    # 存放数据集 [batch_size, sentence_length, 2]
    sentences = []

    # 临时存放一个句子
    sentence = []

    for line in open(path, encoding='UTF-8'):
        line = line.strip()
        if not line:
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(line.split())

    if sentence:
        sentences.append(sentence)

    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    更新为指定标签
    :param sentences:
    :param tag_scheme:
    :return:
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        if not data_utils.check_bio(tags):
            raise Exception('输入的句子应为BIO标注法，请检查%i句' % i)

        if tag_scheme == 'BIO':
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag

        if tag_scheme == 'BIOES':
            new_tags = data_utils.bio_to_bioes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('非法编码')


def word_mapping(sentences):
    """
    构建词典映射
    :param sentences:
    :return:
    """
    word_list = [[x[0] for x in s] for s in sentences]
    dico = data_utils.create_dico(word_list)
    dico['<PAD>'] = 1000001
    dico['<UNK>'] = 1000000
    word_to_id, id_to_word = data_utils.create_mapping(dico)
    return dico, word_to_id, id_to_word


def tag_mapping(sentences):
    """
    构建标签映射
    :param sentences:
    :return:
    """
    tag_list = [[x[-1] for x in s]for s in sentences]
    dico = data_utils.create_dico(tag_list)
    tag_to_id, id_to_tag = data_utils.create_mapping(dico)
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, word_to_id, tag_to_id, train=True):
    """
    数据预处理，返回的list包含：word_list word_id_list word_seg_list, tag_is_list
    :param sentences:
    :param word_to_id:
    :param tag_to_id:
    :param train:
    :return:
    """
    none_index = tag_to_id['O']
    data = []
    for s in sentences:
        word_list = [w[0] for w in s]
        word_id_list = [word_to_id[w if w in word_to_id else '<UNK>'] for w in word_list]
        seg_list = data_utils.get_seg_feature("".join(word_list))
        if train:
            tag_id_list = [tag_to_id[w[-1]] for w in s]
        else:
            tag_id_list = [none_index for w in s]
        data.append([word_list, word_id_list, seg_list, tag_id_list])
    return data


if __name__ == '__main__':
    sentences = load_sentences('./data/ner.dev')
    update_tag_scheme(sentences, 'BIOES')
    _, word_to_id, id_to_word = word_mapping(sentences)
    _, tag_to_id, id_to_tag = tag_mapping(sentences)
    data = prepare_dataset(sentences, word_to_id, tag_to_id)
    print(1)
