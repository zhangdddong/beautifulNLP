#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-08-26 9:00
# @Description: In User Settings Edit
# @Software : PyCharm
import os
import pathlib
import torch


root_path = pathlib.Path(__file__).parent.parent.absolute()

pointer = True
coverage = True
fine_tune = True
train_path = os.path.join(root_path, 'files', 'train.txt')
dev_path = os.path.join(root_path, 'files', 'dev.txt')
test_path = os.path.join(root_path, 'files', 'test.txt')
stop_word_file = os.path.join(root_path, 'files', 'HIT_stop_words.txt')
if pointer:
    if coverage:
        if fine_tune:
            model_name = 'ft_pgn'
        else:
            model_name = 'cov_pgn'
    else:
        model_name = 'pgn'
else:
    model_name = 'baseline'
encoder_save_name = os.path.join(root_path, 'saved_model', model_name, 'encoder.pt')
decoder_save_name = os.path.join(root_path, 'saved_model', model_name, 'decoder.pt')
attention_save_name = os.path.join(root_path, 'saved_model', model_name, 'attention.pt')
reduce_state_save_name = os.path.join(root_path, 'saved_model', model_name, 'reduce_state.pt')
losses_path = os.path.join(root_path, 'saved_model', model_name, 'val_losses.pkl')
log_path = os.path.join(root_path, 'runs', model_name)

is_cuda = False
DEVICE = torch.device('cuda' if is_cuda else 'cpu')
hidden_size = 512
dec_hidden_size = 512
embed_size = 300
max_vocab_size = 20000
embed_file = None
max_src_len = 300
max_tgt_len = 300
truncate_src = True
truncate_tgt = True
min_dec_steps = 30
max_dec_steps = 100
enc_rnn_dropout = 0.5
enc_attention = True
dec_attention = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.001
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 8
batch_size = 8
max_grad_norm = 2.0
LAMBDA = 1

# Beam Search
beam_size = 3
alpha = 0.2
beta = 0.2
gamma = 0.6
