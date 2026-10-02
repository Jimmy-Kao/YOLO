import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        hidden = max(8, channels // ratio)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=[2, 3], keepdim=True)
        max_ = torch.max(x, dim=2, keepdim=True)[0].max(3, keepdim=True)[0]

        avg = self.fc2(F.relu(self.fc1(avg)))
        max_ = self.fc2(F.relu(self.fc1(max_)))

        return x * torch.sigmoid(avg + max_)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        x_cat = torch.cat([avg, max_], dim=1)
        return x * torch.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
