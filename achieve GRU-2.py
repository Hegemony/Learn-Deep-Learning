import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import zipfile

import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_jay_lyrics():
    """加载周杰伦歌词数据集"""
    with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))   # 单词的列表
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])   # 单词和数字组合的字典
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]    # 将歌词的单词转换为数字（由上面的字典定义的）
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()

'''
GRU简洁实现
'''
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

'''
在PyTorch中我们直接调用nn模块中的GRU类即可。
'''
lr = 1e-2  # 注意调整学习率
gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
