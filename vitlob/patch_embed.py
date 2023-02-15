import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from utils import trunc_normal_

class EmbeddingStem(nn.Module):
    def __init__(self, channels=1, embedding_dim=128):
        super(EmbeddingStem, self).__init__()

        num_patches = 100 + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
#         num_patches += 1

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embedding_dim))

        patch_dim = 16
        leaky_alpha = 0.01
        self.projection = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 8, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(16),
        )

        self.linear_projection = nn.Linear(16, embedding_dim)

    def forward(self, x):
        x = self.projection(x)
        x = x.squeeze(3).permute(0,2,1)
#             b, s, f = x.shape
#             x = x.reshape(b, -1, 4 * f)
#             print(x.shape)
        x = self.linear_projection(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        return x
