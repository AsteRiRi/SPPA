import math
from copy import deepcopy
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import utils
import modules
import convnet


class TransNet(modules.Deeplabv3):
    def __init__(self, args, metadata, net_configs, method_configs) -> None:
        super().__init__(args['num_classes'], args['distributed'])
        self.projector = modules.projector.Projector()

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        size = (x.shape[2], x.shape[3])
        out, _, f = self.backbone(x)
        output = self.aspp(f)
        output = self.pre_classifier(output)
        out['embeddings'] = output
        output = self.classifier(output)
        out['logits'] = F.interpolate(output, size=size, mode='bilinear', align_corners=False)

        out['f'] = f
        out['trans_f'] = self.projector(f)

        return out


class TransNet_T(nn.Module):
    def __init__(self, args, metadata, net_configs, method_configs):
        super().__init__()
        self.backbone = convnet.resnet.resnet101()
        self.projector = modules.projector.Projector_T()
        utils.checkpoint.load_model(self, args['best_old_model_path'], metadata, False)
        self.freeze()

    def forward(self, x: Tensor):
        out, _, f = self.backbone(x)
        out['trans_f'], out['recon_f'] = self.projector(f)
        out['f'] = f
        return out

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False



class TransNetTrainer():
    def __init__(self, args, dist_args, metadata, method_configs, net_configs, idx) -> None:
        self.args = args
        self.dist_args = dist_args
        self.method_configs = method_configs
        self.net_configs = net_configs
        self.idx = idx
        self.train_scheme = method_configs['trans_training_scheme']
        self.rank = dist_args['rank'] if dist_args else 0
        if args['distributed']:
            self.train_scheme['batch_size'] = int(math.ceil(
                self.train_scheme['batch_size'] / dist_args['world_size']
            ))
            self.train_scheme['val_batch_size'] = int(math.ceil(
                self.train_scheme['batch_size'] / dist_args['world_size']
            ))
        self.net = TransNet_T(self.args, metadata, net_configs, method_configs)
        self.loss_recon = nn.MSELoss(reduction="mean")
        self.net.to(metadata['device'])
        if args['distributed']:
            self.net = DistributedDataParallel(self.net, device_ids=[metadata['device']])

    def train_transnet(self, datasets, metadata):
        if self.args['distributed']:
            train_sampler = DistributedSampler(
                datasets['train'], self.dist_args['world_size'], self.dist_args['rank'], shuffle=True, seed=self.args['seed'], drop_last=True
            )
        else:
            train_sampler = None

        train_loader = DataLoader(
            datasets['train'],
            batch_size=self.train_scheme['batch_size'],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.args['num_workers'],
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )
        epochs = min(
            self.train_scheme['epochs'],
            math.floor(self.train_scheme['iters'] / len(train_loader))
        )
        self.train_scheme['epochs'] = epochs
        self.train_scheme['iters'] = epochs * len(train_loader)

        self.train_scheme['milestones_iter'] = []
        for item in self.train_scheme['milestones']:
            self.train_scheme['milestones_iter'].append(self.train_scheme['iters'] * item)

        self._init_optimizer_scheduler()

        avg_loss = dict()
        utils.reset_metadata(metadata)
        for epoch in range(self.train_scheme['epochs']):
            if self.args['distributed']:
                train_sampler.set_epoch(epoch)
            for images, _ in train_loader:
                metadata['cur_iter'] += 1
                images = images.to(metadata['device'], dtype=torch.float32)
                loss_dict = self._train(images)
                self.scheduler.step()
                cur_lr = self.optimizer.param_groups[0]['lr']

                if self.rank == 0:
                    self._add_loss(avg_loss, loss_dict)

                if metadata['cur_iter'] % self.args['print_interval'] == 0 and self.rank == 0:
                    train_metrics = {
                        "epoch": epoch,
                        "iteration": iter,
                        "train_loss": self._average_loss(avg_loss, self.args['print_interval'])
                    }
                    scalars_dict = deepcopy(train_metrics['train_loss'])
                    scalars_dict['LR'] = cur_lr
                    metadata['writer'].writer.add_scalars(
                        f"TransNet_Train_{self.idx}",
                        scalars_dict,
                        metadata['cur_iter']
                    )
                    self._clear_loss(avg_loss)

            metadata['cur_epoch'] += 1

        # discard decoder part
        if self.args['distributed']:
            del self.net.module.projector.deconvs
        else:
            del self.net.projector.deconvs
        # Saving network
        print("Saving latest TransNet...")
        utils.checkpoint.save_model(
            self.net.module.projector if self.args['distributed'] else self.net.projector,
            self.net_configs['CKPT_PATH'] + str(self.idx)
        )
        utils.reset_metadata(metadata)

    def _train(self, x: Tensor):
        out = self.net(x)
        if self.method_configs['trans_class_weighted_loss']:
            logits = out['logits']
            loss = self.loss_recon(out['f'], out['recon_f'], logits)
        else:
            loss = self.loss_recon(out['f'], out['recon_f'])
        loss_dict = {
            "loss_f": loss.item()
        }
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return loss_dict

    def _forward(self, x: Tensor):
        with torch.no_grad():
            out = self.net(x)
        loss = self.val_loss(out['f'], out['recon_f'])
        loss_dict = {
            "loss_f": loss.item()
        }
        return loss_dict

    def _add_loss(self, loss_dict, update_dict):
        for k, v in update_dict.items():
            if k not in loss_dict:
                loss_dict[k] = v
            else:
                loss_dict[k] += v

    def _average_loss(self, loss_dict, den):
        for k, v in loss_dict.items():
            loss_dict[k] /= den

        return loss_dict

    def _clear_loss(self, loss_dict):
        loss_dict = dict()

    def _init_optimizer_scheduler(self):
        if self.args['distributed']:
            net = self.net.module
        else:
            net = self.net

        self.optimizer = torch.optim.SGD(
            net.projector.parameters(),
            self.train_scheme['lr'],
            momentum=self.train_scheme['momentum'],
            weight_decay=self.train_scheme['weight_decay']
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            self.train_scheme['milestones_iter'],
            self.train_scheme['gamma']
        )
