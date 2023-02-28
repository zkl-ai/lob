import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
from patch_embed import EmbeddingStem
# from transformer import Transformer
from modules import OutputLayer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class VisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        qkv_bias=False,
        mlp_ratio=2.0,
        dropout_rate=0.1,
        num_classes= 3,
    ):
        super(VisionTransformer, self).__init__()

        # embedding layer
        self.embedding_layer = EmbeddingStem(channels=in_channels, embedding_dim=embedding_dim)

        # transformer
#         self.transformer = Transformer(
#             dim=embedding_dim,
#             depth=num_layers,
#             heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             attn_dropout=dropout_rate,
#             dropout=dropout_rate,
#             qkv_bias=qkv_bias,
#         )
        self.transformer =Transformer(100, 2, 4, 25, 200, 0.0)
#         self.post_transformer_ln = nn.LayerNorm(embedding_dim)
        self.cls_layer = OutputLayer(
            100,#embedding_dim,
            num_classes=num_classes,
        )
#         self.cls_layer = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer(x)
#         x = self.post_transformer_ln(x)
#         x = x[:, 0]
        x = self.cls_layer(x)
        return x