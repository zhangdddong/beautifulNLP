#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-08-19 18:28
# @Description: In User Settings Edit
# @Software : PyCharm
import random
import os
import sys
import pathlib
import torch
import jieba

root_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(root_path)
from model import config
from model.models import PGN
from model.dataset import PairDataset
from model.utils import source2ids, outputids2words, Beam, timer, add2heap, replace_oovs


class Predict(object):
    def __init__(self):
        self.DEVICE = config.DEVICE
        dataset = PairDataset(
            config.train_path,
            max_src_len=config.max_src_len,
            max_tgt_len=config.max_tgt_len,
            truncate_src=config.truncate_src,
            truncate_tgt=config.truncate_tgt
        )
        self.vocab = dataset.build_vocab(embed_file=config.embed_file)
        self.model = PGN(self.vocab)
        with open(config.stop_word_file, encoding='UTF-8') as f:
            self.stop_word = list(set([
                self.vocab[x.strip()] for x in f
            ]))
        self.model.load_model()
        self.model.to(self.DEVICE)

    def greedy_search(self, x, max_sum_len, len_oovs, x_padding_masks):
        """
        :param x: (Tensor) Input sequence as the source.
        :param max_sum_len: (int) The maximum length a summary can have.
        :param len_oovs: (Tensor) Numbers of out-of-vocabulary tokens.
        :param x_padding_masks: (Tensor) The padding masks for the input sequences with shape (batch_size, seq_len).
        :return:
        """
        encoder_output, encoder_states = self.model.encoder(replace_oovs(x, self.vocab))
        decoder_states = self.model.reduce_state(encoder_states)
        x_t = torch.ones(1) * self.vocab.SOS
        x_t = x_t.to(self.DEVICE, dtype=torch.int64)
        summary = [self.vocab.SOS]
        coverage_vector = torch.zeros([1, x.shape[1]]).to(self.DEVICE)
        while int(x_t.item()) != self.vocab.EOS and len(summary) < max_sum_len:
            covtext_vector, attention_weights, coverage_vector = self.model.attention(
                decoder_states,
                encoder_output,
                x_padding_masks,
                coverage_vector
            )
            p_vocab, decoder_states, p_gen = self.model.decoder(
                x_t.unsqueeze(1),
                decoder_states,
                covtext_vector
            )
            final_dist = self.model.get_final_distribution(
                x,
                p_gen,
                p_vocab,
                attention_weights,
                torch.max(len_oovs)
            )
            x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)
            decoder_word_idx = x_t.item()
            summary.append(decoder_word_idx)
            x_t = replace_oovs(x_t, self.vocab)

        return summary

    def best_k(self, beam, k, encoder_output, x_padding_masks, x, len_oovs):
        x_t = torch.tensor(beam.tokens[-1]).reshape(1, 1)
        x_t = x_t.to(self.DEVICE)
        context_vector, attention_weights, coverage_vector = self.model.attention(
            beam.decoder_states,
            encoder_output,
            x_padding_masks,
            beam.coverage_vector
        )
        p_vocab, decoder_states, p_gen = self.model.decoder(
            replace_oovs(x_t, self.vocab),
            beam.decoder_states,
            context_vector
        )
        final_dist = self.model.get_final_distribution(
            x,
            p_gen,
            p_vocab,
            attention_weights,
            torch.max(len_oovs)
        )
        log_probs = torch.log(final_dist.squeeze())
        if len(beam.tokens) == 1:
            forbidden_ids = [
                self.vocab[u'这'],
                self.vocab[u'此'],
                self.vocab[u'采用'],
                self.vocab[u'，'],
                self.vocab[u'。']
            ]
            log_probs[forbidden_ids] = -float('inf')
        log_probs[self.vocab.EOS] *= config.gamma * x.size()[1] / len(beam.tokens)
        log_probs[self.vocab.UNK] = -float('inf')
        topk_probs, topk_idx = torch.topk(log_probs, k)
        best_k = [
            beam.extend(x, log_probs[x], decoder_states, coverage_vector) for x in topk_idx.tolist()
        ]
        return best_k

    def beam_search(self, x, max_sum_len, beam_width, len_oovs, x_padding_masks):
        encoder_output, encoder_states = self.model.encoder(replace_oovs(x, self.vocab))
        coverage_vector = torch.zeros([1, x.shape[1]]).to(self.DEVICE)
        decoder_states = self.model.reduce_state(encoder_states)

        attention_weights = torch.zeros([1, x.shape[1]]).to(self.DEVICE)
        init_beam = Beam(
            [self.vocab.SOS],
            [0],
            decoder_states,
            coverage_vector
        )
        k = beam_width
        curr, completed = [init_beam], []

        for _ in range(max_sum_len):
            topk = []
            for beam in curr:
                if beam.tokens[-1] == self.vocab.EOS:
                    completed.append(beam)
                    k -= 1
                    continue
                for can in self.best_k(beam, k, encoder_output, x_padding_masks, x, torch.max(len_oovs)):
                    add2heap(topk, (can.seq_score(), id(can), can), k)
            curr = [items[2] for items in topk]
            if len(completed) == beam_width:
                break
        completed += curr
        result = sorted(completed, key=lambda x: x.seq_score(), reverse=True)[0].tokens
        return result

    def predict(self, text, tokenize=True, beam_search=True):
        if isinstance(text, str) and tokenize:
            text = list(jieba.cut(text))
        x, oov = source2ids(text, self.vocab)
        x = torch.tensor(x).to(self.DEVICE)
        len_oovs = torch.tensor([len(oov)]).to(self.DEVICE)
        x_padding_masks = torch.ne(x, 0).byte().float()
        if beam_search:
            summary = self.beam_search(
                x.unsqueeze(0),
                max_sum_len=config.max_dec_steps,
                beam_width=config.beam_size,
                len_oovs=len_oovs,
                x_padding_masks=x_padding_masks
            )
        else:
            summary = self.greedy_search(
                x.unsqueeze(0),
                max_sum_len=config.max_dec_steps,
                len_oovs=len_oovs,
                x_padding_masks=x_padding_masks
            )
        summary = outputids2words(summary, oov, self.vocab)
        return summary.replace('<SOS>', '').replace('<EOS>', '').strip()


if __name__ == '__main__':
    pred = Predict()
    print('vocab_size: ', len(pred.vocab))
    with open(config.test_path, 'r', encoding='UTF-8') as f:
        picked = random.choice(list(f))
        source, ref = picked.strip().split('<sep>')

    greedy_prediction = pred.predict(source.split(), beam_search=False)
    beam_prediction = pred.predict(source.split(), beam_search=True)
    print('source: ', source)
    print('ref: ', ref)
    print('greedy: ', greedy_prediction)
    print('beam: ', beam_prediction)
