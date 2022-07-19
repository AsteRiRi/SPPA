from typing import Union, List, Dict
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss
import torchvision.transforms.functional as trans_F

from torch.nn.parameter import Parameter
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode



class UnbiasedCELoss(nn.Module):
    def __init__(self, num_old_cls: int=None, reduction: str='mean', ignore_index: int=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.num_old_cls = num_old_cls

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        outputs = torch.zeros_like(inputs)
        den = torch.logsumexp(inputs, dim=1)
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:self.num_old_cls], dim=1) - den
        outputs[:, self.num_old_cls:] = inputs[:, self.num_old_cls:] - den.unsqueeze(dim=1)
        labels = labels.clone()
        labels[labels < self.num_old_cls] = 0

        return F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)



class UnbiasedKDLoss(nn.Module):
    def __init__(self, factor: float=1.0, reduction: str='mean', alpha: float=0.5):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.factor = factor

    def forward(self, inputs: Tensor, targets: Tensor, mask: Tensor=None) -> Tensor:
        num_cls = inputs.shape[1]
        num_old_cls = targets.shape[1]
        device = inputs.device

        targets = targets * self.alpha
        inputs = inputs * self.alpha
        den = torch.logsumexp(inputs, dim=1)
        index_new = torch.tensor(
            [0] + [i for i in range(num_old_cls, num_cls)],
            dtype=torch.long, device=device
        )
        inputs_new = torch.index_select(inputs, index=index_new, dim=1)
        p_bkg = torch.logsumexp(inputs_new, dim=1) - den
        p_inputs_old = inputs[:, 1:num_old_cls] - den.unsqueeze(dim=1)
        targets = torch.softmax(targets, dim=1)
        loss = - (targets[:, 0] * p_bkg + (targets[:, 1:] * p_inputs_old).sum(dim=1))
        loss = loss / num_old_cls

        if mask is not None:
            loss = loss * mask.float()

        loss = (1 / self.alpha)**2 * loss

        if self.reduction == 'mean':
            return self.factor * torch.mean(loss)
        elif self.reduction == 'sum':
            return self.factor * torch.sum(loss)
        elif self.reduction == 'none':
            return self.factor * loss
        else:
            raise NotImplementedError



class FeatureTransferLoss(nn.Module):
    """
    Feature alignment loss for projector

    Args:
        factor: The weight of this loss
        reduction: Specifies the reduction to apply

    Inputs:
        f1, f2: The representations from old and new projectors, respectively
    """
    def __init__(
        self, factor: float=1.0, reduction: str="mean"
    ):
        super().__init__()
        self.factor = factor
        self.criterion = nn.L1Loss(reduction=reduction)
       
    def forward(self, f1: Tensor, f2: Tensor) -> Tensor:
        f1 = F.normalize(f1, dim=1, p=2)
        f2 = F.normalize(f2, dim=1, p=2)
        loss = self.criterion(f1, f2)

        return self.factor * loss



class FeatureClusteringLoss(nn.Module):
    """
    Feature compacting loss in contrastive learning

    Args:
        factor: The weight of this loss
        no_bkg: If ignore background class

    Inputs:
        features: Feature map of current network, with shape of B * C * H * W
        labels: GT of current input image, with shape of B * H * W
        prototypes: A list of prototypes for each class
    
    """

    def __init__(
        self, factor: float=1.0, no_bkg: bool=False
    ):
        super().__init__()

        self.factor = factor
        self.no_bkg = no_bkg

    def forward(self, features: Tensor, labels: Tensor, prototypes: Tensor):
        device = features.device
        b, c, h, w = features.shape
        labels = trans_F.resize(labels, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        cls_new = torch.unique(labels)
        if self.no_bkg:
            cls_new = cls_new[1:]
        if cls_new[-1] == 255:
            cls_new = cls_new[:-1]

        loss = torch.tensor(0., device=device)
        criterion = nn.MSELoss()
        features = features.permute(0, 2, 3, 1).reshape(-1, c)
        labels = labels.permute(0, 2, 3, 1).reshape(-1, 1).squeeze()
        for cl in cls_new:
            features_cl = features[(labels == cl), :]
            loss += criterion(
                features_cl,
                prototypes[cl].unsqueeze(0).expand(features_cl.shape[0], -1)
            )
        loss /= cls_new.shape[0]

        return self.factor * loss



class FeatureSeperationLoss(nn.Module):
    """
    Feature seperation part in contrastive learning

    Args:
        factor: The weight of this loss
        margin: The minimal margin between class prototypes
        num_class: Number of classes
        no_bkg: If ignore background class

    Inputs:
        features: feature map of current network, with shape of B * C * H * W
        labels: GT of current input image, with shape of B * H * W
        prototypes: a list of prototypes for each class
    """

    def __init__(
        self, factor: float=1.0, margin: float=0., num_class: int=0, no_bkg: bool=False
    ):
        super().__init__()

        self.factor = factor
        self.margin = margin
        self.no_bkg = no_bkg


    def forward(self, features: Tensor, labels: Tensor, prototypes: Tensor):
        device = features.device
        b, c, h, w = features.shape
        # Avoid changing prototypes
        prototypes = copy.deepcopy(prototypes)
        labels = trans_F.resize(labels, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        cls_new = torch.unique(labels)
        if self.no_bkg:
            cls_new = cls_new[1:]
        if cls_new[-1] == 255:
            cls_new = cls_new[:-1]

        features_local_mean = torch.zeros((self.num_class, c), device=device)
        features = features.permute(0, 2, 3, 1).reshape(-1, c)
        labels = labels.permute(0, 2, 3, 1).reshape(-1, 1).squeeze()
        for cl in cls_new:
            features_cl = features[(labels == cl), :]
            features_local_mean[cl] = torch.mean(features_cl, dim=0)
        features_local_mean_reduced = features_local_mean[cls_new,:]

        REF = features_local_mean_reduced

        REF = F.normalize(REF, p=2, dim=1)
        features_local_mean_reduced = F.normalize(features_local_mean_reduced, p=2, dim=1)
        D = 1 - torch.mm(features_local_mean_reduced, REF.T)
        for i in range(D.shape[0]):
            D[i][i] = 2.0
        loss = torch.mean(torch.clamp(self.margin - D, min=0.0))

        return self.factor * loss



class IntraClassLoss(nn.Module):
    """
    This loss transfer feature relations within the same class, only applied on
    old classes

    Args:
        factor: The weight of this loss

    Inputs:
        features: Feature map of current network, with shape of B * C * H * W
        features_old: Feature map of old network, same shape as `features`
        output_old: Output logits of old net, with shape of B * C_o * H * W
        labels: GT of current input image, with shape of B * H * W

    """
    def __init__(self, factor: float=1.0):
        super().__init__()
        self.factor = factor

    def forward(
            self, features: Tensor, features_old: Tensor,
            outputs_old: Tensor, labels: Tensor, prototypes: Tensor,
            num_old_class: int
        ):
        loss = torch.tensor(0., device=features.device)
        b, c, h, w = features.shape
        labels = trans_F.resize(labels, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        mask = (labels < num_old_class).long()
        pseudo = torch.argmax(outputs_old, dim=1)
        pseudo = trans_F.resize(pseudo, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        pseudo = pseudo * mask
        cls_old = torch.unique(pseudo)
        if cls_old[0] == 0:
            cls_old = cls_old[1:]

        features = features.permute(0, 2, 3, 1).reshape(-1, c)
        features_old = features_old.permute(0, 2, 3, 1).reshape(-1, c)
        pseudo = pseudo.permute(0, 2, 3, 1).reshape(-1, 1).squeeze()
        for cl in cls_old:
            features_cl = features[(pseudo == cl), :]
            features_cl_old = features_old[(pseudo == cl), :]
            prototype_cl = torch.mean(features_cl, dim=0).detach()
            prototype_cl_old = torch.mean(features_cl_old, dim=0).detach()

            criterion = nn.MSELoss(reduction='sum')
            loss_cl = criterion(
                features_cl - prototype_cl,
                features_cl_old - prototype_cl_old
            ) / features_cl.shape[0]

            loss += loss_cl

        if cls_old.shape[0] > 0:
            loss /= cls_old.shape[0]
            return self.factor * loss
        else:
            return torch.tensor(0., device=features.device)



class InterClassLoss(nn.Module):
    """
    This loss transfer prototype relations of different classes in feature space

    Args:
        factor: The weight of this loss

    Inputs:
        features: Feature map of current network, with shape of B * C * H * W
        features_old: Feature map of old network, same shape as `features`
        output_old: output logits of old net, with shape of B * C_o * H * W
        labels: GT of current input image, with shape of B * H * W
    """
    def __init__(self, factor: float=1.0):
        super().__init__()
        self.factor = factor

    def forward(
        self, features: Tensor, features_old: Tensor,
        outputs_old: Tensor, labels: Tensor, prototypes: Tensor,
        num_old_class: int
    ):
        b, c, h, w = features.shape
        labels = trans_F.resize(labels, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        mask = (labels < num_old_class).long()
        pseudo = torch.argmax(outputs_old, dim=1)
        pseudo = trans_F.resize(pseudo, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        pseudo = pseudo * mask

        PROTO_REF = []
        PROTO_CUR = []

        for i in range(b):
            features_i = features[i].permute(1, 2, 0).reshape(-1, c)
            features_old_i = features_old[i].permute(1, 2, 0).reshape(-1, c)
            pseudo_i = pseudo[i].permute(1, 2, 0).reshape(-1, 1).squeeze()
            cls_old = torch.unique(pseudo_i)
            if cls_old[0] == 0:
                cls_old = cls_old[1:]
            for cl in cls_old:
                features_i_cl = features_i[(pseudo_i == cl), :]
                features_old_i_cl = features_old_i[(pseudo_i == cl), :]
                PROTO_CUR.append(torch.mean(features_i_cl, dim=0))
                PROTO_REF.append(torch.mean(features_old_i_cl, dim=0))

        if len(PROTO_REF) == 0:
            return torch.tensor(0., device=features.device)

        PROTO_REF = torch.stack(PROTO_REF, dim=0)
        PROTO_CUR = torch.stack(PROTO_CUR, dim=0)
        
        PROTO_REF = F.normalize(PROTO_REF, p=2, dim=1)
        PROTO_CUR = F.normalize(PROTO_CUR, p=2, dim=1)
        D_REF = torch.mm(PROTO_REF, PROTO_REF.T)
        D_CUR = torch.mm(PROTO_CUR, PROTO_CUR.T)

        criterion = nn.MSELoss()
        loss = criterion(D_CUR, D_REF)

        return self.factor * loss



class WeightedReconLoss(nn.Module):
    """
    This loss add more weight on positions that have old classes
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, f1: Tensor, f2: Tensor, logits_old: Tensor):
        b, c, h, w = f1.shape
        probs = F.softmax(logits_old, dim=1)
        _, pseudo = probs.max(dim=1)
        entropy = self.__entropy(probs)
        pseudo = trans_F.resize(pseudo, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        entropy = trans_F.resize(entropy, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        mask = (pseudo > 0).long()
        weight_map = mask * (1 - entropy) + torch.ones_like(entropy, device=f1.device)
        weight_map = weight_map.expand(-1, f1.shape[1], -1, -1)
        loss = self.criterion(f1, f2) * weight_map
        loss = torch.mean(loss)
        return loss


    def __entropy(self, probs: Tensor) -> Tensor:
        EPS = 1e-8
        C = probs.shape[1]
        return (-1 / math.log(C)) * torch.sum(probs * torch.log(probs + EPS), dim=1)
