from typing import List, Dict, Any

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from convnet import get_convnet
import convnet


__all__ = ["DeepLabV3_256"]


class Deeplabv3(nn.Module):
    def __init__(
        self,
        num_classes: int,
        distributed: bool,
    ) -> None:
        super(Deeplabv3, self).__init__()

        aspp_dilation = [6, 12, 18]
        self.BN_op = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
        self.backbone = convnet.resnet.resnet101()
        self.aspp = ASPP(self.backbone.outplanes, aspp_dilation, distributed)
        self.pre_classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            self.BN_op(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(256, num_classes, 1, 1)

        self.__init_weight(self.pre_classifier)
        self.__init_weight(self.classifier)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        size = (x.shape[2], x.shape[3])
        out, _, f = self.backbone(x)
        output = self.aspp(f)
        output = self.pre_classifier(output)
        out['embeddings'] = output
        output = self.classifier(output)
        out['logits'] = F.interpolate(output, size=size, mode='bilinear', align_corners=False)

        return out

    def __init_weight(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels, BN_op):
        super(ASPPPooling, self).__init__()
        self.module_list = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            BN_op(out_channels),
            nn.ReLU(inplace=True)
        ])

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        for mod in self.module_list:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, BN_op):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            BN_op(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPP(nn.Module):
    def __init__(
        self, in_channels: int, atrous_rates: List[int],
        distributed: bool = False, out_channels: int=256
    ):
        super(ASPP, self).__init__()
        self.BN_op = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
        modules = []
        # the 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            self.BN_op(out_channels),
            nn.ReLU(inplace=True)
        ))
        # the 3x3 atrous conv with rate 6, 12, 18
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, self.BN_op))
        # the image pooling
        modules.append(ASPPPooling(in_channels, out_channels, self.BN_op))

        self.convs = nn.ModuleList(modules)
        self.projector = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            self.BN_op(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.projector(res)
