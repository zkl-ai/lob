import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from utils import trunc_normal_

class EmbeddingStem(nn.Module):
    def __init__(self, channels=1, embedding_dim=128):
        super(EmbeddingStem, self).__init__()

        patch_dim = embedding_dim
        leaky_alpha = 0.01
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, patch_dim, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(patch_dim),
            nn.Conv2d(patch_dim, patch_dim, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(patch_dim),
            nn.Conv2d(patch_dim, patch_dim, kernel_size=(1,10), stride=(1,10)),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(patch_dim),
            nn.Conv2d(patch_dim, patch_dim, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(patch_dim),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(patch_dim, patch_dim, kernel_size=(1,2)),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(patch_dim),
            nn.Conv2d(patch_dim, patch_dim, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=leaky_alpha),
            nn.BatchNorm2d(patch_dim),
        )

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.squeeze(x) # [Batch, Channel, Sequence]
        # x = x.unfold(dimension=-1, size=10, step=10) # [bs x nvars x patch_num x patch_len]
        # x = x.permute(0,2,1)
        # x = x.reshape(x.shape[0], x.shape[1], -1)
        return x
