from typing import Callable, Tuple, List
import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class Projector_T(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.convs = []
        self.convs.append(nn.Sequential(
            nn.Conv2d(2048, 512, 1, stride=1, dilation=1, padding=0),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        ))
        self.convs.append(nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, dilation=2, padding=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        ))
        self.convs.append(nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, dilation=2, padding=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        ))

        self.convs = nn.ModuleList(self.convs)

        self.deconvs = []
        self.deconvs.append(nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        ))
        self.deconvs.append(nn.Sequential(
            nn.Conv2d(1024, 2048, 3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        ))
        self.deconvs = nn.ModuleList(self.deconvs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        for conv in self.convs:
            x = conv(x)
        projected_feature = x
        for deconv in self.deconvs:
            x = deconv(x)

        return projected_feature, x


class Projector(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_op = nn.Conv2d

        self.convs = []
        self.convs = []
        self.convs.append(nn.Sequential(
            self.conv_op(2048, 512, 1, stride=1, dilation=1, padding=0),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        ))
        self.convs.append(nn.Sequential(
            self.conv_op(512, 512, 3, stride=1, dilation=2, padding=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        ))
        self.convs.append(nn.Sequential(
            self.conv_op(512, 512, 3, stride=1, dilation=2, padding=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        ))

        self.convs = nn.ModuleList(self.convs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x)

        return x

