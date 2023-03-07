import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
from patch_embed import EmbeddingStem
from transformer import Transformer
from modules import OutputLayer


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
        self.embedding_layer = EmbeddingStem(channels=in_channels, embedding_dim=embedding_dim,)


        # encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout_rate, batch_first=True)
        # encoder_norm = nn.LayerNorm(d_model)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        d_model = embedding_dim
        # transformer
        self.transformer = Transformer(
            dim=d_model,
            depth=num_layers,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=dropout_rate,
            dropout=dropout_rate,
            qkv_bias=qkv_bias,
        )
        # self.transformer =Transformer(100, 2, 4, 25, 200, 0.0)
        # self.transformer =Transformer(64, 2, 4, 16, 128, 0.0)
        self.cls_layer = OutputLayer(
            d_model,
            num_classes=num_classes,
        )

    def forward(self, x):
        x = self.embedding_layer(x)                                              # x: [bs * nvars x patch_num x d_model]
        x = self.transformer(x)                                                  # z: [bs * nvars x patch_num x d_model]
        x = self.cls_layer(x)
        return x