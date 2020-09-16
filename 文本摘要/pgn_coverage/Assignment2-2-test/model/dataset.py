#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-08-26 10:57
# @Description: In User Settings Edit
# @Software : PyCharm
import sys
import os
import pathlib
from collections import Counter
import torch
from torch.utils.data import Dataset

root_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(root_path)
from model.utils import simple_tokenizer, count_words, source2ids, abstract2ids, sort_batch_by_len
from model.vocab import Vocab
from model import config


class PairDataset(object):
    def __init__(self,
                 filename,
                 tokenize=simple_tokenizer,
                 max_src_len=None,
                 max_tgt_len=None,
                 truncate_src=False,
                 truncate_tgt=False):
        print('Reading dataset %s...' % filename, end=' ', flush=True)
        self.filename = filename
        self.pairs = []

        with open(filename, 'rt', encoding='UTF-8') as f:
            next(f)
            for i, line in enumerate(f):
                pair = line.strip().split('<sep>')
                if len(pair) != 2:
                    print('Line %d of %s is malformed.' % (i, filename))
                    print(line)
                    continue
                src = tokenize(pair[0])
                if max_src_len and len(src) > max_src_len:
                    if truncate_src:
                        src = src[:max_src_len]
                    else:
                        continue
                tgt = tokenize(pair[1])
                if max_tgt_len and len(tgt) > max_tgt_len:
                    if truncate_tgt:
                        tgt = tgt[:max_tgt_len]
                    else:
                        continue
                self.pairs.append((src, tgt))
        print('%d pairs.' % len(self.pairs))

    def build_vocab(self, embed_file):
        word_counts = Counter()
        count_words(word_counts, [src + tgt for src, tgt in self.pairs])
        vocab = Vocab()
        for word, count in word_counts.most_common(config.max_vocab_size):
            vocab.add_words([word])
        if embed_file is not None:
            count = vocab.load_embeddings(embed_file)
            print('%d pre-trained embeddings loaded.' % count)
        return vocab


class SampleDataset(Dataset):
    def __init__(self, data_pair, vocab):
        self.src_sents = [x[0] for x in data_pair]
        self.tgt_sents = [x[1] for x in data_pair]
        self.vocab = vocab
        self._len = len(data_pair)

    def __getitem__(self, index):
        x, oov = source2ids(self.src_sents[index], self.vocab)
        res = {
            'x': [self.vocab.SOS] + x + [self.vocab.EOS],
            'OOV': oov,
            'len_OOV': len(oov),
            'y': abstract2ids(self.tgt_sents[index], self.vocab, oov),
            'x_len': len(self.src_sents[index]),
            'y_len': len(self.tgt_sents[index])
        }
        return res

    def __len__(self):
        return self._len


def collate_fn(batch):
    def padding(indice, max_length, pad_idx=0):
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    data_batch = sort_batch_by_len(batch)

    x = data_batch["x"]
    x_max_length = max([len(t) for t in x])
    y = data_batch["y"]
    y_max_length = max([len(t) for t in y])

    OOV = data_batch["OOV"]
    len_OOV = torch.tensor(data_batch["len_OOV"])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)

    x_len = torch.tensor(data_batch["x_len"])
    y_len = torch.tensor(data_batch["y_len"])
    return x_padded, y_padded, x_len, y_len, OOV, len_OOV
