import torch
import torch.nn as nn
import torch.nn.functional as F

class APAM(nn.Module):
    def __init__(self, c1, c2):
        super(APAM, self).__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1)
        self.bn1 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c2, c2, 3, padding=1, groups=c2)  # depthwise
        self.residual = c1 == c2  # 判斷是否能相加

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        if self.residual:
            return x + out
        else:
            return out
