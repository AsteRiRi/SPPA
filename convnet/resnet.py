"""
Adapted from ResNet Pytorch implementation for Deeplab
"""
import os
from typing import Dict, Type, Any, Callable, Union, List, Optional, Tuple

import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F


BASE_PATH = './pretrained/'
model_paths = {
    'resnet101': BASE_PATH + 'resnet101.pth',
}


def conv3x3(
        in_planes: int, out_planes: int,
        stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class HeadLayer(nn.Module):

    def __init__(self, BN_ops=nn.BatchNorm2d):
        super(HeadLayer, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BN_ops(32)
        self.relu = nn.ReLU(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BN_ops(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = BN_ops(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.maxpool(out)

        return out
    

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:

        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity

        return out
        

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:

        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        groups: int=1,
        width_per_group: int=64,
        output_stride: int=16,
        zero_init_residual: bool=False,
        ret_features: bool=False,
        feature_with_relu: bool=False,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:

        super(ResNet, self).__init__()
        self.ret_features = ret_features
        self.feature_with_relu = feature_with_relu
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 64
        self.outplanes = 512 * block.expansion
        self.lowlevel_planes = 64 * block.expansion

        if output_stride == 16:
            self.dilation = [1, 1, 1, 2]
        elif output_stride == 8:
            self.dilation = [1, 1, 2, 4]
        else:
            raise NotImplementedError(f"output stride {output_stride} is not supported")

        self.stem_layer = HeadLayer(self._norm_layer)
    
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=1,
            dilation=self.dilation[0]
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            dilation=self.dilation[1]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilation=self.dilation[2]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilation=self.dilation[3]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the 
        # residual branch starts with zeros, and each residual block behaves 
        # like an identity. This improves the model by 0.2~0.3% 
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        out = dict()
        if self.ret_features:
            x = self.stem_layer(x)
            x = self.layer1(x)
            out['layer1'] = x
            x = F.relu(x, inplace=self.feature_with_relu)
            lowlevel_feature = x
            x = self.layer2(x)
            out['layer2'] = x
            x = F.relu(x, inplace=self.feature_with_relu)
            x = self.layer3(x)
            out['layer3'] = x
            x = F.relu(x, inplace=self.feature_with_relu)
            x = self.layer4(x)
            out['layer4'] = x
            x = F.relu(x, inplace=self.feature_with_relu)

        else:
            x = self.stem_layer(x)
            x = self.layer1(x)
            x = F.relu(x, inplace=True)
            lowlevel_feature = x
            x = self.layer2(x)
            x = F.relu(x, inplace=True)
            x = self.layer3(x)
            x = F.relu(x, inplace=True)
            x = self.layer4(x)
            x = F.relu(x, inplace=True)

        return out, lowlevel_feature, x

    def _make_layer(
        self, block: Type[Union[BasicBlock, Bottleneck]], 
        planes: int, blocks: int, stride: int = 1, dilation: int = 1
    ) -> nn.Sequential:

        layers = []
        norm_layer = self._norm_layer

        if dilation != 1:
            stride = 1

        multi_grid = [1, 2, 4]
        first_dilation = multi_grid[0] * dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride != 1:
                downsample = nn.Sequential(
                    nn.AvgPool2d(2, stride=2),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    norm_layer(planes * block.expansion),
                )
            elif dilation != 1:
                downsample = nn.Sequential(
                    nn.Identity(),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
        else:
            downsample = None

        layers.append(block(
            self.inplanes, planes, stride, self.groups,
            self.base_width, first_dilation, downsample, norm_layer
        ))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width,
                dilation=dilation if multi_grid is None else multi_grid[i] * dilation,
                norm_layer=norm_layer
            ))

        return nn.Sequential(*layers)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    output_stride: int,
    sync_BN: bool,
    ret_features: bool,
    feature_with_relu: bool,
    **kwargs: Any
) -> ResNet:

    norm_layer = nn.SyncBatchNorm if sync_BN else nn.BatchNorm2d
    model = ResNet(
        block, layers, norm_layer=norm_layer, output_stride=output_stride,
        ret_features=ret_features, feature_with_relu = feature_with_relu, **kwargs
    )

    if pretrained:
        state_dict = _load_state_dict_from_file(model_paths[arch])
        del state_dict['fc.weight'], state_dict['fc.bias']
        model.load_state_dict(state_dict)
        del state_dict
    return model


def _load_state_dict_from_file(path, device='cpu'):
    assert os.path.exists(path)
    state_dict = torch.load(path, map_location=device)
    return state_dict


def resnet101(**kwargs: Any) -> ResNet:
    return _resnet(
        'resnet101', Bottleneck, [3, 4, 23, 3], True, 16, True, True, True, **kwargs
    )
