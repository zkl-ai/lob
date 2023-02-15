import torch.nn as nn

from patch_embed import EmbeddingStem
from transformer import Transformer
from modules import OutputLayer

   
class VisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        embedding_dim=128,
        num_layers=2,
        num_heads=8,
        qkv_bias=False,
        mlp_ratio=2.0,
        dropout_rate=0.1,
        num_classes= 3,
    ):
        super(VisionTransformer, self).__init__()

        # embedding layer
        self.embedding_layer = EmbeddingStem(channels=in_channels, embedding_dim=embedding_dim)

        # transformer
        self.transformer = Transformer(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=dropout_rate,
            dropout=dropout_rate,
            qkv_bias=qkv_bias,
        )
        self.post_transformer_ln = nn.LayerNorm(embedding_dim)
        self.cls_layer = OutputLayer(
            embedding_dim,
            num_classes=num_classes,
        )
#         self.cls_layer = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer(x)
        x = self.post_transformer_ln(x)
#         x = x[:, 0]
        x = self.cls_layer(x)
        return x