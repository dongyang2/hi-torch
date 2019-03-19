# 第八节
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {}/{} = {:.4f}'.format(len(keep_words), len(self.word2index),
                                                 len(keep_words)/len(self.word2index)))

        # reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        for word in keep_words:
            self.add_word(word)


def normalize_string(s):
    """Lowercase and remove non-letter characters"""
    s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def index_from_sentence(voc, sentence):
    """Takes string sentence, returns sentence of word indexes"""
    # 这里好神奇，pycharm知道我这里的voc是Voc，直接voc. 会出来很多成员变量
    return [voc.word2index[word] for word in sentence.split(' ')]+[EOS_token]


if __name__ == '__main__':
    import os
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as fun
    import re
    import unicodedata
    import numpy as np
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    print('-'*15, 'Start', time.ctime(), '-'*15, '\n')

    device = torch.device("cpu")
    MAX_LENGTH = 10  # Maximum sentence length

    # Default word tokens
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    print('%s%s %s %s %s' % ('\n', '-'*16, 'End', time.ctime(), '-'*16))
