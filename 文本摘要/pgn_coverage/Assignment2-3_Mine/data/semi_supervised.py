#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-09-03 15:46
# @Description: In User Settings Edit
# @Software : PyCharm
import pathlib
import sys
root_path = pathlib.Path(__file__).parent
sys.path.append('..')
from models.predict import Predict
from data.data_utils import write_samples


def semi_supervised(samples_path, write_path, beam_search):
    pred = Predict()
    semi = []
    with open(samples_path, 'r') as f:
        for line in f:
            source, ref = line.strip().split('<sep>')
            prediction = pred.predict(ref.split(), beam_search=beam_search)
            semi.append(prediction + '<sep>' + ref)
    write_samples(semi, write_path)


if __name__ == '__main__':
    samples_path = 'output/train.txt'
    write_path_greedy = 'output/semi_greedy.txt'
    write_path_beam = 'output/semi_beam.txt'
    beam_search = False
    if beam_search:
        write_path = write_path_beam
    else:
        write_path = write_path_greedy
    semi_supervised(samples_path, write_path, beam_search)
