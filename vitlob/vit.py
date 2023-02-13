import torch.nn as nn

from patch_embed import EmbeddingStem
from transformer import Transformer
from modules import OutputLayer

class ResConv2d(nn.Module):
    """
    Mainly for convenience - combination of convolutional layer with zero padding and a leaky ReLU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, leaky_alpha=0.01):
        super(ResConv2d, self).__init__()
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.layer_block  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding),
            nn.LeakyReLU(leaky_alpha)
        )

    def forward(self, x):
        return self.layer_block(x)

class ResBlock(nn.Module):
    """
    The residual block used in the DeepResLOB model, architecture is as per our report.
    """

    def __init__(self, n_filters, num_layers=3, leaky_alpha=0.01, kernel_sizes=None):
        super(ResBlock, self).__init__()

        if kernel_sizes is None:
            self.kernel_sizes = [(3,1) for i in range(num_layers)]
        else:
            self.kernel_sizes = kernel_sizes
        
        layers = [ResConv2d(n_filters, n_filters, self.kernel_sizes[i], leaky_alpha) for i in range(num_layers)]
        self.res_block = nn.Sequential(
            *layers
        )
    
    def forward(self, x):
        residual = self.res_block(x)

        return residual + x
  
   
class VisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        embedding_dim=64,
        num_layers=4,
        num_heads=8,
        qkv_bias=False,
        mlp_ratio=4.0,
        use_revised_ffn=False,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        use_conv_stem=False,
        use_conv_patch=False,
        use_linear_patch=True,
        use_conv_stem_original=False,
        use_stem_scaled_relu=False,
        hidden_dims=None,
        cls_head=True,
        num_classes= 3,
        representation_size=16, #None,
    ):
        super(VisionTransformer, self).__init__()
#         res_filters=16
#         res_layers=3
#         res_blocks=2
#         leaky_alpha=0.01
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, res_filters, kernel_size=(1,2), stride=(1,2)),
#             nn.LeakyReLU(leaky_alpha)
#         )

#         res_layers1 = [ResBlock(res_filters, res_layers, leaky_alpha) for i in range(res_blocks)]

#         self.res_block1 = nn.Sequential(
#             *res_layers1
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(res_filters, res_filters, kernel_size=(1,2), stride=(1,2)),
#             nn.LeakyReLU(leaky_alpha)
#         )

#         res_layers2 = [ResBlock(res_filters, res_layers, leaky_alpha) for i in range(res_blocks)]

#         self.res_block2 = nn.Sequential(
#             *res_layers2
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv2d(res_filters, res_filters, kernel_size=(1,10)),
#             nn.LeakyReLU(leaky_alpha)
#         )

#         res_layers3 = [ResBlock(res_filters, res_layers, leaky_alpha) for i in range(res_blocks)]

#         self.res_block3 = nn.Sequential(
#             *res_layers3
#         )

        # embedding layer
        self.embedding_layer = EmbeddingStem(
            channels=in_channels,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            conv_patch=use_conv_patch,
            linear_patch=use_linear_patch,
            conv_stem=use_conv_stem,
            conv_stem_original=use_conv_stem_original,
            conv_stem_scaled_relu=use_stem_scaled_relu,
            position_embedding_dropout=dropout_rate,
            cls_head=cls_head,
        )

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
#         x = self.res_block1(self.conv1(x))
#         x = self.res_block2(self.conv2(x))#[32, 16, 100, 10]
#         x = self.res_block3(self.conv3(x))#[32, 16, 100, 1]
# #         x = x.squeeze(3).permute(0,2,1)
#         x = x.permute(0,2,1,3)
#         x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.embedding_layer(x)
        x = self.transformer(x)
        x = self.post_transformer_ln(x)
        x = self.cls_layer(x)
        return x