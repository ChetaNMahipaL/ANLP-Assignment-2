import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import numpy as np
import torch.optim as optim
from train import device
    
#<----------------------------------------------Feed Forward Neural Network------------------------------------------------------>

class FeedForward(nn.Module):
    def __init__(self, model_dim, hid_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(model_dim, hid_dim)
        self.ac1 = nn.ReLU().to(device)
        self.l2 = nn.Linear(hid_dim, model_dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self, inp):
        inp = self.l1(inp)
        inp = self.dropout(self.ac1(inp))
        inp = self.l2(inp)
        return inp
    
#<--------------------------------------------MultiHead Attention---------------------------------------------------------------->

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dim_key = self.model_dim // self.num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear layers for query, key, and value
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)

    def attention_val(self, Q, K, V, mask=None):
        score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_key)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        attn_weight = self.dropout(torch.softmax(score, dim=-1))

        new_val = torch.matmul(attn_weight, V)
        return new_val

    def split_layers(self, x):
        batch_size, seq_len, model_dim = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.dim_key).transpose(1, 2)

    def combine_layers(self, x):
        batch_size, _, seq_len, dim_key = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)

    def forward(self, Q, K, V, mask=None):
        # Split into multiple heads
        Q = self.split_layers(self.query(Q))
        K = self.split_layers(self.key(K))
        V = self.split_layers(self.value(V))

        layer_out = self.attention_val(Q, K, V, mask)
        final_output = self.out(self.combine_layers(layer_out))

        return final_output

#<---------------------------------------------------------Encoder--------------------------------------------------------------->

class Encoder(nn.Module):
    def __init__(self, model_dim, num_heads, hid_dim, dropout):
        super(Encoder, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads,dropout)
        self.norm = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, hid_dim,dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, mask):

        att_score = self.self_attn(inp, inp, inp, mask)
        inp = self.norm(inp + self.dropout(att_score))
        ffn_out = self.ffn(inp).to(device)
        inp = self.norm(inp + self.dropout(ffn_out))

        return inp