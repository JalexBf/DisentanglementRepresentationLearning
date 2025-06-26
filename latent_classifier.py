import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentClassifier(nn.Module):
    def __init__(self, z_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.shape = nn.Linear(128, 3)
        self.scale = nn.Linear(128, 6)
        self.orient = nn.Linear(128, 40)
        self.posX = nn.Linear(128, 32)
        self.posY = nn.Linear(128, 32)

    def forward(self, z):
        h = self.net(z)
        return [
            self.shape(h),
            self.scale(h),
            self.orient(h),
            self.posX(h),
            self.posY(h),
        ]
