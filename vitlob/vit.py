import torch.nn as nn

from patch_embed import EmbeddingStem
from transformer import Transformer
from modules import OutputLayer

   
class VisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        embedding_dim=64,
        num_layers=2,
        num_heads=8,
        qkv_bias=False,
        mlp_ratio=4.0,
        use_revised_ffn=False,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        cls_head=True,
        num_classes= 3,
        representation_size=None,
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
            attn_dropout=attn_dropout_rate,
            dropout=dropout_rate,
            qkv_bias=qkv_bias,
            revised=use_revised_ffn,
        )
        self.post_transformer_ln = nn.LayerNorm(embedding_dim)

        # output layer
        self.cls_layer = OutputLayer(
            embedding_dim,
            num_classes=num_classes,
            representation_size=representation_size,
            cls_head=cls_head,
        )

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer(x)
        x = self.post_transformer_ln(x)
        x = self.cls_layer(x)
        return x