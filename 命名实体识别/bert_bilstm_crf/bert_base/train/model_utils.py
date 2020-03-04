#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import collections
import os
import codecs
import pickle
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

from bert_base.bert import modeling, tokenization
from bert_base.train.lstm_crf_layer import BLSTM_CRF
from bert_base.train.train_helper import set_logging


logger = set_logging('Enter model utils')


class InputExample(object):
    def __init__(self, guid=None, text=None, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, dropout_rate=1.0,
                 lstm_size=1, cell='lstm', num_layers=1):
    """
    创建模型
    :param bert_config: bert配置
    :param is_training: 是否训练
    :param input_ids: 数据的idx表示
    :param input_mask:
    :param segment_ids:
    :param labels: 标签的idx表示
    :param num_labels: 类别数量
    :param use_one_hot_embeddings:
    :param dropout_rate:
    :param lstm_size:
    :param cell:
    :param num_layers:
    :return:
    """
    # 使用数据加载BertModel，获取对应的字embedding
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    # 获取对应的embedding，输入数据[batch_size, seq_length, embedding_size]
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value

    # 序列的真实长度
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size]大小的向量，包含了当前的batch中的序列长度

    # 添加CRF output layer
    blstm_crf = BLSTM_CRF(
        embedd_chars=embedding,
        hidden_unit=lstm_size,
        cell_type=cell,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        initializers=initializers,
        num_labels=num_labels,
        seq_length=max_seq_length,
        labels=labels,
        lengths=lengths,
        is_training=is_training
    )

    res = blstm_crf.add_blstm_crf_layer(crf_only=True)
    return res


def write_tokens(tokens, output_dir, mode):
    """
    将序列解析结果写入到文件中，只是在mode==test的时候使用
    :param tokens:
    :param output_dir:
    :param mode:
    :return:
    """
    if mode == 'test':
        path = os.path.join(output_dir, 'token_' + mode + '.txt')
        f = codecs.open(path, 'a', encoding='UTF-8')
        for token in tokens:
            if token != '**NULL**':
                f.write(token + '\n')
        f.close()


def convert_single_examples(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode):
    """
    将一个样本进行分析，然后将字转化成id，标签转化成id，然后结构化到InputFeature对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param output_dir:
    :param mode:
    :return:
    """
    label_map = {}
    # 从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index的map
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as f:
            pickle.dump(label_map, f)

    text_list = example.text.split(' ')
    label_list = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(text_list):
        # 分词，如果是中文，就是分字，但是对于一些不在Bert的vocab.txt中得字符会被进行WordPice处理（例如中文中得引号）
        # 可以将所有得分词操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_tmp = label_list[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_tmp)
            else:
                # 一般不会出现else情况
                labels.append('X')
        # 序列截断
        if len(tokens) >= max_seq_length - 1:
            # -2得原因是因为序列需要加一个句首和句尾标志
            tokens = tokens[0: (max_seq_length - 2)]
            labels = labels[0: (max_seq_length - 2)]

        n_tokens = []
        segment_ids = []
        label_ids = []

        # -----句始添加CLS标志-----
        n_tokens.append('[CLS]')
        segment_ids.append(0)
        label_ids.append(label_map['CLS'])

        for token_i, token in enumerate(tokens):
            n_tokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])

        # -----句尾添加SEP标志-----
        n_tokens.append('[SEP]')
        segment_ids.append(0)
        label_ids.append(label_map['[SEP]'])

        input_ids = tokenizer.convert_tokens_to_ids(n_tokens)   # 将序列中得字转换成ID形式
        input_mask = [1] * len(input_ids)

        # padding 使用
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            n_tokens.append('**NULL**')

        # 判断长度
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        # 打印部分样本信息
        if ex_index < 5:
            logger.info('*** Example ***')
            logger.info('guid: %s' % example.guid)
            logger.info('tokens: %s' % ' '.join([tokenization.printable_text(x) for x in tokens]))
            logger.info('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
            logger.info('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
            logger.info('segment_ids: %s' % ' '.join([str(x) for x in segment_ids]))
            logger.info('label_ids: %s' % ' '.join([str(x) for x in label_ids]))

        # 结构化一个类
        feature = InputFeature(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=label_ids
        )

        # mode='test'的时候才有效
        write_tokens(n_tokens, output_dir, mode)

        return feature


def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                             output_file, output_dir, mode=None):
    """
    将数据存储成TF_Record格式。将数据转换为TF_Record结构，作为模型数据输入。
    :param examples: 样本 [InputExample1, InputExample2, ... ]
    :param label_list: 标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer对象
    :param output_file: 输出的文件名字
    :param output_dir: 输入的文件路径
    :param mode: mode
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info('Writing example %d of %d' % (ex_index, len(examples)))
        # 对于每一个训练样本
        feature = convert_single_examples(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode)

        # 转换整型feature
        def create_int_feature(values):
            f =tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features['input_ids'] = create_int_feature(feature.input_ids)
        features['input_mask'] = create_int_feature(feature.input_mask)
        feature['segment_ids'] = create_int_feature(feature.segment_ids)
        feature['label_ids'] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """
    将TF_Record文件读取出来
    :param input_file: TF_Record文件
    :param seq_length: 序列的长度
    :param is_training: 是否训练
    :param drop_remainder:
    :return:
    """
    name_to_features = {
        'input_ids': tf.FixedLenFeature([seq_length], tf.int64),
        'input_mask': tf.FixedLenFeature([seq_length], tf.int64),
        'segment_ids': tf.FixedLenFeature([seq_length], tf.int64),
        'label_ids': tf.FixedLenFeature([seq_length], tf.int64)
    }

    def _decode_record(record, name_to_features):
        """
        :param record:
        :param name_to_features:
        :return:
        """
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        """
        :param params: dict类型，包含各种参数
        :return:
        """
        batch_size = params['batch_size']
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_calls=2,   # 并行处理数据的CPU核心数量，不要大于机器的核心数量
                drop_remainder=drop_remainder
            )
        )
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn


# --------------------------------------
