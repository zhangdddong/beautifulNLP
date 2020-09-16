#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-08-19 20:16
# @Description: In User Settings Edit
# @Software : PyCharm
import sys
import os
import pathlib
from rouge import Rouge
import jieba

root_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(root_path)
from model.predict import Predict
from model.utils import timer
from model import config


class RougeEval(object):
    def __init__(self, path):
        self.path = path
        self.scores = None
        self.rouge = Rouge()
        self.sources = []
        self.hypos = []
        self.refs = []
        self.process()

    def process(self):
        print('Reading from ', self.path)
        with open(self.path, 'r', encoding='UTF-8') as f:
            for line in f:
                source, ref = line.strip().split('<sep>')
                ref = ''.join(list(jieba.cut(ref))).replace('。', '.')
                self.sources.append(source)
                self.refs.append(ref)
        print('Test set contains {} samples'.format(len(self.sources)))

    def build_hypos(self, predict):
        print('Building hypotheses')
        count = 0
        for source in self.sources:
            count += 1
            if count % 100 == 0:
                print(count)
            self.hypos.append(predict.predict(source.split()))

    def get_average(self):
        assert len(self.hypos) > 0, 'Build hypotheses first!'
        print('Calculating average rouge scores.')
        return self.rouge.get_scores(self.hypos, self.refs, avg=True)

    def one_sample(self, hypo, ref):
        return self.rouge.get_scores(hypo, ref)[0]


if __name__ == '__main__':
    rouge_eval = RougeEval(config.test_path)
    predict = Predict()
    rouge_eval.build_hypos(predict)
    result = rouge_eval.get_average()
    print('rouge1: ', result['rouge-1'])
    print('rouge2: ', result['rouge-2'])
    print('rougeL: ', result['rouge-l'])
    with open('../files/rouge_result.txt', 'a') as file:
        for r, metrics in result.items():
            file.write(r + '\n')
            for metric, value in metrics.items():
                file.write(metric + ': ' + str(value * 100))
                file.write('\n')
