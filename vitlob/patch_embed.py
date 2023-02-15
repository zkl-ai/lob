import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from utils import trunc_normal_

class EmbeddingStem(nn.Module):
    def __init__(self, channels=1, embedding_dim=128):
        super(EmbeddingStem, self).__init__()

        num_patches = 100x
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        num_patches += 1

        # positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embedding_dim)
        )
        self.pos_drop = nn.Dropout(p=position_embedding_dropout)

        patch_dim = 32,#32,#16
        leaky_alpha = 0.01
        self.projection = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(leaky_alpha),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(leaky_alpha),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(1,10)),
            nn.LeakyReLU(leaky_alpha),
            nn.BatchNorm2d(64),
#                 nn.Linear(patch_dim, embedding_dim),
        )

#             self.linear_projection = nn.Linear(8*4, embedding_dim)

    def forward(self, x):
        x = self.projection(x)
        x = x.squeeze(3).permute(0,2,1)
#             b, s, f = x.shape
#             x = x.reshape(b, -1, 4 * f)
#             print(x.shape)
#         x = self.linear_projection(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        return self.pos_drop(x + self.pos_embed)