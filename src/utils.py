import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

#<--------------------------------- Preparing and Processing Data ----------------------------------->
def retrieve_corpus(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        corpus = []
        for line in file:
            # print(line.strip().lower())
            corpus.append(line.lower())
        return corpus

def remove_punctuation(tokenized_sentence):
    return [word for word in tokenized_sentence if word not in string.punctuation]

def create_freq(word_freq, data):

    for _, sentence in enumerate(data):
        for word in sentence:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    return word_freq

def create_unk_emb(word_freq, data, threshold):

    for i, sentence in enumerate(data):
        for j,word in enumerate(sentence):
            if word_freq[word] < threshold:
                data[i][j] = "<unk>"
    return data

def create_vocab(word_freq, threshold):

    vocab = ["<pad>", "<sos>", "<eos>", "<unk>"]
    for word in word_freq:
        if word_freq[word]>=threshold:
            vocab.append(word)
    return vocab

def word2idx_fun (vocab):
    word2idx = {}
    idx2word = {}
    for i, word in enumerate(vocab):
        word2idx[word] = i
        idx2word[i] = word
    return word2idx, idx2word

def prepare_data(sentences, word2idx,max_len=100):
    for i, sentence in enumerate(sentences):

        if len(sentence) > max_len-2:
            sentences[i] = sentence[:max_len-2]
            sentences[i] = ["<sos>"] + sentences[i] + ["<eos>"]
        else:
            sentences[i] = ["<sos>"] + sentences[i] + ["<eos>"]
    
    all_indices = []
    for sentence in sentences:
        index = []
        for word in sentence:
            if word in word2idx:
                index.append(word2idx[word])
            else:
                index.append(word2idx["<unk>"])

        index = index + [word2idx["<pad>"]] * (max_len-len(index))
        index = torch.tensor(index)
        all_indices.append(index)  
        
    return all_indices

#<----------------------------Dataset Loader----------------------------------------->

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = self.src_data[idx]
        tgt_input = self.tgt_data[idx]

        return src, tgt_input

#<---------------------------------------------Positional Encoding------------------------------------------------------------->

class PosEncoding(nn.Module):
    def __init__(self, model_dim, max_len):
        super(PosEncoding, self).__init__()
        pos_code = torch.zeros(max_len, model_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        scale = torch.exp(torch.arange(0, model_dim, 2, dtype=torch.float) *
                          -(math.log(10000.0) / model_dim))

        pos_code[:, 0::2] = torch.sin(pos * scale)
        pos_code[:, 1::2] = torch.cos(pos * scale)

        self.register_buffer('pos_code', pos_code.unsqueeze(0))

    def forward(self, inp):
        seq_len = inp.size(1)
        inp = inp + self.pos_code[:, :seq_len]

        return inp