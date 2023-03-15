# coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F


class FLOB(nn.Module):
    def __init__(self):
        super().__init__()
        patch_dim = 64
        leaky_alpha = 0.01
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        # self.seq1 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 1)),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.BatchNorm2d(16),
        # )
        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 10), stride=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        # self.seq2 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 1), stride=(2, 1)),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.BatchNorm2d(32),

        # )
        self.res2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        # self.seq3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(3, 1)),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.BatchNorm2d(64),
        # )
        # self.seq4 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(8, 1)),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.BatchNorm2d(64),
        # )

        self.res3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        self.seq = nn.Sequential(
            nn.Linear(100, 64),
            nn.LeakyReLU(leaky_alpha),
            nn.Linear(64, 32),
            nn.LeakyReLU(leaky_alpha),
            nn.Linear(32, 16),
            nn.LeakyReLU(leaky_alpha),
            nn.Linear(16, 8),
            nn.LeakyReLU(leaky_alpha),
            nn.Linear(8, 1),
            nn.LeakyReLU(leaky_alpha),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(64, 3),
            )

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.res1(x)
        # x = self.seq1(x)
        x = self.conv2(x)
        x = x + self.res2(x)
        # x = self.seq2(x)
        x = self.conv3(x)
        x = x + self.res3(x)
        # x = self.seq3(x)
        # x = self.seq4(x)
        x = x.squeeze()
        x = self.seq(x)
        x = x.squeeze()
        x = self.fc1(x)
        return x
