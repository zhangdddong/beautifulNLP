#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# __author__ = 'zd'
import os
import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import timedelta


MAX_VOCAB_SIZE = 10000      # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'     # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic)})
    return vocab_dic


def build_dataset(config, use_word):
    if use_word:
        tokenizer = lambda x: x.split(' ')      # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]    # char-level

    if os.path.exists(config.vocab_path):
        vocab = pickle.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pickle.dump(vocab, open(config.vocab_path, 'wb'))
    print(f'Vocab size: {len(vocab)}')

    def load_dataset(path):
        """
        :param path:
        :return: [([...], 2, 100), ([...], 5, 150), ...]    ([单词id, 分类标签, 句子长度])
        """
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, lable = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(lable), seq_len))
        return contents
    train = load_dataset(config.train_path)
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.text_path)
    return vocab, train, dev, test


def build_net_data(dataset, config):
    data = [x[0] for x in dataset]
    data_x = pad_sequences(data, maxlen=config.max_len)
    label_y = [x[1] for x in dataset]
    label_y = tf.keras.utils.to_categorical(label_y, num_classes=config.num_classes)
    return data_x, label_y


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    train_dir = './data/dataset/test.txt'
    vocab_dir = './data/embedding/vocab.pkl'
    pretrain_dir = './data/embedding/sgns.sogou.char.bz2'
    emb_dim = 300
    filename_trimmed_dir = './data/embedding/embedding_SougouNews'

    if os.path.exists(vocab_dir):
        word_to_id = pickle.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')      # 以单词为单位构建词表
        tokenizer = lambda x: [y for y in x]      # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pickle.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    with open(pretrain_dir, encoding='UTF-8') as f:
        for i, line in enumerate(f):
            lin = line.strip().split(' ')
            if lin[0] in word_to_id:
                idx = word_to_id[lin[0]]
                emb = [float(x) for x in lin[1: 301]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
