import torch
import torch.nn as nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super(Attention, self).__init__()
        assert (
            dim % num_heads == 0
        ), "Embedding dimension should be divisible by number of heads"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout_rate=0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class OutputLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes=3,
    ):
        super(OutputLayer, self).__init__()

        self.num_classes = num_classes

        self.net = nn.Linear(embedding_dim, num_classes)

        self.num_classes = num_classes

    def forward(self, x):
        # x = x[:, 0]
        x = x.mean(dim = 1)
        return self.net(x)