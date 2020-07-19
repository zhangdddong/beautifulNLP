import numpy as np
import copy
import time
import tensorflow as tf
import pickle
import collections

def batch_generator(data, batch_size, n_steps):
    data = copy.copy(data)
    batch_steps = batch_size * n_steps
    n_batches = int(len(data) / batch_steps)
    data = data[:batch_size * n_batches]
    data = data.reshape((batch_size, -1))
    while True:
        np.random.shuffle(data)
        for n in range(0, data.shape[1], n_steps):
            x = data[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y


class Vocabulary(object):
    def __init__(self):
        self.vocab = ["unk"]

    def load_vocab(self, vocabfile, limit_size=5000):
        assert os.path.exists(vocabfile)
        with open(vocabfile, 'rb') as f:
            self.vocab += pickle.load(f)[0:limit_size]

    def build_vocab(self, data, limit_size=5000):
        counter = collections.Counter(data)
        word_freq = counter.most_common(limit_size)
        self.vocab, freq = zip(*word_freq)

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word2id(self, word):
        if word in self.vocab:
            return self.vocab.index(word)
        else:
            return 0

    def id2word(self, idx):
        if index < len(self.vocab):
            return self.vocab[idx]
        else:
            raise Exception('Unknown index!')

    def encode(self, text):
        arr = []
        for word in text.split():
            arr.append(self.word2id(word))
        return np.array(arr)

    def decode(self, arr):
        words = []
        for index in arr:
            words.append(self.id2word(index))
        return "".join(words)

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.vocab, f)
