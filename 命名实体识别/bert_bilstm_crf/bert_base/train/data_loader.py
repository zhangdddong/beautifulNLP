#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import codecs
import os
import pickle

from bert_base.bert import tokenization
from bert_base.train.model_utils import InputExample


class DataProcessor(object):
    """
    数据处理类
    """
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """
        读取BIO数据
        :param input_file:
        :return:
        """
        with codecs.open(input_file, 'r', encoding='UTF-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contents = line.strip()
                tokens = contents.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contents) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in word if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contents.startswith('-DOCSTART-'):
                    words.append('')
                    continue
            return lines


class NerProcessor(DataProcessor):
    def __init__(self, output_dir):
        self.labels = set()
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        pass

    def get_dev_examples(self, data_dir):
        pass

    def get_test_examples(self, data_dir):
        pass

    def get_labels(self, labels=None):
        """
        获取数据中的标签数据
        :return:
        """
        if labels is not None:
            try:
                # 从支持的文件中读取标签类型
                if os.path.exists(labels) and os.path.isfile(labels):
                    with codecs.open(labels, 'r', encoding='UTF-8') as f:
                        for line in f:
                            self.labels.add(line.strip())
                else:
                    # 否则通过传入的参数，按照逗号分割
                    self.labels = labels.split(',')
                # to set
                self.labels = set(self.labels)
            except Exception as e:
                print(e)

        # 通过读取train文件获取标签的方法会出现一定风险
        if os.path.exists(os.path.join(self.output_dir, 'label_list.pkl')):
            with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'rb') as f:
                self.labels = pickle.load(f)
        else:
            if len(self.labels) > 0:
                self.labels = self.labels.union(set(['X', '[CLS]', '[SEP]']))
                with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'wb') as f:
                    pickle.dump(self.labels, f)
            else:
                self.labels = [
                    'O',
                    'B-TIM',
                    'B-TIM',
                    'I-TIM',
                    'B-PER',
                    'I-PER',
                    'B-ORG',
                    'I-ORG',
                    'B-LOC',
                    'I-LOC',
                    'X',
                    '[CLS]',
                    '[SEP]'
                ]
        return self.labels

    def _create_examples(self, lines, set_type):
        """
        :param lines: 二维矩阵 [['O B-PER O'], ['char1, char2, char3'], ...]
        :param set_type: 类型
        :return: list [InputExample1, InputExample2, ... ]
        """
        examples = []
        for i, line in enumerate(lines):
            guid = '%s-%s' % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_data(self, input_file):
        """
        读取BIO数据
        :param input_file:
        :return:
        """
        with codecs.open(input_file, 'r', encoding='UTF-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contents = line.strip()
                tokens = contents.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                else:
                    if len(contents) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                self.labels.add(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
                        continue
                if contents.startswith('-DOCSTART-'):
                    continue
            return lines

