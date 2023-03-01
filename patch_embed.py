import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from utils import trunc_normal_
class EmbeddingStem(nn.Module):
    def __init__(self, channels=1, embedding_dim=128):
        super(EmbeddingStem, self).__init__()

        num_patches = 32 + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
#         num_patches += 1

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embedding_dim))

#         patch_dim = 16
#         leaky_alpha = 0.01
#         self.projection = nn.Sequential(
#             nn.Conv2d(1, 2, kernel_size=(1,2), stride=(1,2)),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(2),
#             nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(3,1), padding='same'),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(2),
#             nn.Conv2d(2, 8, kernel_size=(1,2), stride=(1,2)),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(8),
#             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,1), padding='same'),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(8),
#             nn.Conv2d(8, 32, kernel_size=(1,10)),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,1), padding='same'),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(32),
#         )
#         patch_dim = 32
#         leaky_alpha = 0.01
#         self.projection = nn.Sequential(
#             nn.Conv2d(1, patch_dim, kernel_size=(1,2), stride=(1,2)),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(patch_dim),
#             nn.Conv2d(in_channels=patch_dim, out_channels=patch_dim, kernel_size=(4,1), padding='same'),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(patch_dim),
#             nn.Conv2d(patch_dim, patch_dim, kernel_size=(1,2), stride=(1,2)),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(patch_dim),
#             nn.Conv2d(in_channels=patch_dim, out_channels=patch_dim, kernel_size=(4,1), padding='same'),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(patch_dim),
#             nn.Conv2d(patch_dim, patch_dim, kernel_size=(1,10)),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(patch_dim),
#             nn.Conv2d(in_channels=patch_dim, out_channels=patch_dim, kernel_size=(4,1), padding='same'),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(patch_dim),
#         )
        patch_dim = 32
        leaky_alpha = 0.01
        self.projection = nn.Sequential(
            nn.Conv2d(1, patch_dim, kernel_size=(3,4), stride=(1,4), padding=(1,0)),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(patch_dim),
#             nn.Conv2d(in_channels=patch_dim, out_channels=patch_dim, kernel_size=(4,1), padding='same'),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(patch_dim),
            nn.Conv2d(patch_dim, patch_dim, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(patch_dim),
#             nn.Conv2d(in_channels=patch_dim, out_channels=patch_dim, kernel_size=(4,1), padding='same'),
#             nn.LeakyReLU(negative_slope=leaky_alpha),
#             nn.BatchNorm2d(patch_dim),
        )
#         in_channels=1
# #         gru_units=64
#         res_filters=32
# #         inception_filters=32
#         res_layers=1
#         res_blocks=1
#         leaky_alpha=0.01

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, res_filters, kernel_size=(1,2), stride=(1,2)),
#             nn.LeakyReLU(leaky_alpha),
#             nn.BatchNorm2d(res_filters)
#         )

#         res_layers1 = [ResBlock(res_filters, res_layers, leaky_alpha) for i in range(res_blocks)]

#         self.res_block1 = nn.Sequential(
#             *res_layers1
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(res_filters, res_filters, kernel_size=(1,2), stride=(1,2)),
#             nn.LeakyReLU(leaky_alpha),
#             nn.BatchNorm2d(res_filters)
            
#         )

#         res_layers2 = [ResBlock(res_filters, res_layers, leaky_alpha) for i in range(res_blocks)]

#         self.res_block2 = nn.Sequential(
#             *res_layers2
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv2d(res_filters, res_filters, kernel_size=(1,10)),
#             nn.LeakyReLU(leaky_alpha),
#             nn.BatchNorm2d(res_filters)
#         )

#         res_layers3 = [ResBlock(res_filters, res_layers, leaky_alpha) for i in range(res_blocks)]

#         self.res_block3 = nn.Sequential(
#             *res_layers3
#         )



    def forward(self, x):
#         x = self.res_block1(self.conv1(x))
#         x = self.res_block2(self.conv2(x))
#         x = self.res_block3(self.conv3(x))
        x = self.projection(x)
        x = torch.squeeze(x)
#         x = x.squeeze(3).permute(0,2,1)
#             b, s, f = x.shape
#             x = x.reshape(b, -1, 4 * f)
#             print(x.shape)
#         x = self.linear_projection(x)
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_token, x), dim=1)
#         x = x + self.pos_embed
        return x
