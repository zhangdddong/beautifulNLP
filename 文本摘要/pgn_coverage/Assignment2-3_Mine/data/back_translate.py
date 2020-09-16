#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-09-03 15:12
# @Description: In User Settings Edit
# @Software : PyCharm
# pip3 install googletrans 参考: https://blog.csdn.net/MacwinWin/article/details/105183415
import jieba
import os
import time
from tqdm import tqdm
from googletrans import Translator
from data.data_utils import write_samples

translator = Translator(service_urls=['translate.google.cn'])


def translate(q, source, target):
    result = translator.translate(q, dest=target).text
    return result


def back_translate(q):
    en = translate(q, 'zh-CN', 'en')
    target = translate(en, 'en', 'zh-CN')
    return target


def translate_continue(sample_path, translate_path):
    translated = []
    count = 0
    with open(sample_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            count += 1
            time.sleep(2)
            source, ref = tuple(line.strip().split('<sep>'))
            source = back_translate(source.strip())
            ref = back_translate(ref.strip())
            source = ' '.join(list(jieba.cut(source)))
            ref = ' '.join(list(jieba.cut(ref)))
            translated.append(source + '<sep>' + ref)
    write_samples(translated, translate_path)


if __name__ == '__main__':
    sample_path = 'output/train.txt'
    translate_path = 'output/translated.txt'
    translate_continue(sample_path, translate_path)
