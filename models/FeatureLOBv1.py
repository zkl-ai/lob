# coding=utf8
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

class TimeBlock(nn.Module):
    def __init__(self, idx, embed_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(100, embed_dim),
            nn.ReLU(),
        )
        self.idx = idx

    def forward(self, x): # [batch, feature, seq]
        x = x[:, self.idx:(self.idx+1), :]
        z = self.fc1(x)
        return z


class TimeEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.timeblocks = nn.ModuleList([TimeBlock(i, embed_dim) for i in range(40)])

    def forward(self, x): # [batch, feature, seq]
        z = None
        for tb in self.timeblocks:
            t = tb(x)
            if z is None:
                z = t
            else:
                z = torch.cat((z,t), dim=1)
        return z

class FLOB(nn.Module):
    def __init__(self):
        super().__init__()
        num_heads = 8
        embed_dim = 128
        self.time_embedding  = TimeEmbed(embed_dim)
        dim_feedforward = int(embed_dim * 2)
        dropout_rate = 0.0
        num_layers = 2
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout_rate, batch_first=True)
        encoder_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        # self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # self.attention = Transformer(embed_dim, 1, 2, 16, 64, 0.0)
        self.fc1 = nn.Sequential(
            nn.Linear(40, 3),
        )

    def forward(self, x): #(batch, 1, seq, feature)
        # x = torch.squeeze(x)
        x = x.permute(0, 1, 3, 2)       #(batch, 1, feature, seq)
        x = self.time_embedding(x)      #(batch, 1, feature, seq_feature)
        # x = x.permute(0, 1, 3, 2)     #(batch, 1, seq, feature)
        x = torch.squeeze(x)            #(batch, feature, seq)
        # x, _ = self.attention(x, x, x)
        x = self.attention(x)
        # x = x.mean(dim = 1)
        x = x[:, :, 0]
        x = self.fc1(x)
        return x
