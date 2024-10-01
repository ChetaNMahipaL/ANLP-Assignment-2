import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import numpy as np
import torch.optim as optim
from utils import PosEncoding
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, vocab_en, vocab_fr, model_dim, num_layer, num_heads, hid_dim, max_len, device, dropout=0.1):
        super(Transformer, self).__init__()

        self.vocab_en = vocab_en
        self.vocab_fr = vocab_fr
        self.model_dim = model_dim
        self.embed_en = nn.Embedding(len(vocab_en), model_dim)
        self.embed_fr = nn.Embedding(len(vocab_fr), model_dim)

        self.pos_enc = PosEncoding(model_dim, max_len)
        self.encoders = nn.ModuleList([Encoder(model_dim, num_heads, hid_dim, dropout, device) for _ in range(num_layer)])
        self.decoders = nn.ModuleList([Decoder(model_dim, num_heads, hid_dim, dropout, device) for _ in range(num_layer)])
        self.final_layer = nn.Linear(model_dim, len(vocab_fr))
        self.max_len = max_len
        self.device = device

    def decode(self, src, max_len=40):
        batch_size = src.size(0)

        src_emb = self.pos_enc(self.embed_en(src))
        for encoder in self.encoders:
            src_emb = encoder(src_emb, None)

        tgt = torch.ones(batch_size, 1).long().to(self.device)


        for _ in range(max_len):

            src_mask, tgt_mask = self.generate_square_subsequent_mask(src, tgt)
            tgt_emb = self.pos_enc(self.embed_fr(tgt))
            for decoder in self.decoders:
                tgt_emb = decoder(tgt_emb, src_emb, src_mask, tgt_mask)

            output = self.final_layer(tgt_emb)
            next_word = output[:, -1:, :]
            next_word = torch.argmax(next_word, dim=-1)
            tgt = torch.cat((tgt, next_word), dim=1)
            # tgt = out_labels

        return tgt


    def forward(self, src, target):
        src = src
        src_mask, target_mask = self.generate_square_subsequent_mask(src, target)
        src_mask = src_mask
        target_mask = target_mask
        src_emb = self.pos_enc(self.embed_en(src))
        target_emb = self.pos_enc(self.embed_fr(target))

        for encoder in self.encoders:
            src_emb = encoder(src_emb, src_mask)

        for decoder in self.decoders:
            target_emb = decoder(target_emb, src_emb, src_mask, target_mask)

        return self.final_layer(target_emb)

    def generate_square_subsequent_mask(self,src, target):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (target != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt_mask.size(2)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(self.device)
        return src_mask, tgt_mask