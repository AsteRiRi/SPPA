from typing import Iterator, Tuple, Dict, Any

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data as data


def get_optimizer(
        model_parameters: Iterator[nn.parameter.Parameter],
        args: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Tuple[optim.Optimizer, lr_scheduler._LRScheduler]:
    cur_step_idx = args['steps'].index(metadata['cur_step'])
    optimizer = optim.SGD(
        model_parameters, 
        args['lr'][cur_step_idx],
        momentum=args['momentum'],
        weight_decay=args['weight_decay']
    )

    scheduler = PolyLR(
        optimizer,
        metadata['train_iters'],
        args['lr_power']
    )

    return optimizer, scheduler


class PolyLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, min_lr=1e-6, last_epoch=-1):
        self.power = power
        self.max_iters = max_iters + 1
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch/self.max_iters)**self.power, self.min_lr)
                for base_lr in self.base_lrs]


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
    model.train()


def chk_freeze(model: nn.Module) -> bool:
    return all(not param.requires_grad for param in model.parameters())
