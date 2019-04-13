# 第八节
import torch
import torch.nn as nn
import torch.nn.functional as fun


class Voc:
    def __init__(self, name):  # 这里的init里没有super，是不是因为这个类是我们自创的不是继承的所以不用写
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


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layer=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.embedding = embedding
        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layer, dropout=(0 if n_layer == 1 else dropout),
                          bidirectional=True)

    def forward(self, input_seq, input_length, hidden=None):
        embed = self.embedding(input_seq)  # Convert word indexes to embeddings
        packed = nn.utils.rnn.pack_padded_sequence(embed, input_length)  # Pack padded batch of sequences for RNN module
        output, hidden = self.gru(packed, hidden)  # Forward pass through GRU
        output, _ = nn.utils.rnn.pad_packed_sequence(output)  # Unpack padding
        output = output[:, :, :self.hidden_size]+output[:, :, self.hidden_size:]  # Sum bidirectional GRU outputs
        # Return output and final hidden state
        return output, hidden


class Attn(nn.Module):
    """Attention layer"""
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, 'is not an appropriate attention method.')
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size*2, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden*encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden*energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        # 下面这个式子肯定是想逼死我
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v*energy, dim=2)

    def forward(self, hidden, encoder_output):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'dot':
            attn_energy = self.dot_score(hidden, encoder_output)
        elif self.method == 'general':
            attn_energy = self.general_score(hidden, encoder_output)
        else:
            attn_energy = self.concat_score(hidden, encoder_output)

        # Transpose max_length and batch_size dimensions
        attn_energy = attn_energy.t()

        # Return the softmax normalized probability scores (with added dimension)
        return fun.softmax(attn_energy, dim=1).unsqueeze(1)


if __name__ == '__main__':
    import os
    import time
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
