import math
import os
from typing import Dict, Tuple, Any
from copy import deepcopy

import torch
import torch.cuda.amp as amp
from torch import Tensor
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
import torchvision.transforms.functional as trans_F
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from .base import BaseMethod
import utils
import modules
import models


class Method(BaseMethod):
    
    def __init__(self, args, dist_args, metadata) -> None:
        super().__init__(args, dist_args, metadata)

        self.logger = utils.logger.R0Logger(__name__, self.rank)
        
        # Default method configs
        self.method_configs = {
            "label_type": "pseudo",
            "pseudo_per_class": True,
            "pseudo_entropy_min": 0.1,
            "pseudo_entropy_max": 0.5,
            "pseudo_percent": 0.8,
            "alpha": 0.5,
            "weight_ce": 1.0,
            "weight_kd": 5.0,
            "weight_intra": 5e-2,
            "weight_inter": 10.0,
            "weight_fc": 1e-2,
            "weight_fs": 1e-2,
            "weight_ft": 30,
            "proto_layer": "embeddings",
            "proto_channels": 256,
            "trans_training_scheme": {
                "batch_size": 32,
                "val_batch_size": 32,
                "epochs": 40,
                "iters": 2000,
                "milestones": [0.8],
                "gamma": 0.2,
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4
            }
        }
        self.net_configs = {
            "CKPT_PATH": args['logging_path'] + "/trans_"         
        }
        self.load_configs(args, metadata)

        self.net = self.init_net(metadata)
        self.old_net = None

        if self.method_configs['label_type'] in ['pseudo', 'gt']:
            self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=255)
        #self.loss_kd = utils.loss.UnbiasedKDLoss(
        #    alpha = self.method_configs['alpha'],
        #    factor = self.method_configs['weight_kd']
        #)
        self.loss_fc = utils.loss.FeatureClusteringLoss(
            factor=self.method_configs['weight_fc'],
            no_bkg=args['overlap'],
            normalize=False
        )
        self.loss_fs = None
        self.loss_intra = utils.loss.IntraClassLoss(
            self.method_configs['weight_intra']
        )
        self.loss_inter = utils.loss.InterClassLoss(
            self.method_configs['weight_inter']
        )
        self.loss_ft = utils.loss.FeatureTransferLoss(
            self.method_configs['weight_ft'], reduction="mean"
        )

        self.method_vars = [
            'prototypes', 'count_features'
        ]
        self.prototypes = torch.zeros(
            (args['num_classes'], self.method_configs['proto_channels']),
            device=metadata['device'],
            requires_grad=False
        )
        self.count_features = torch.zeros(
            (args['num_classes'], ),
            device=metadata['device'],
            requires_grad=False
        )

    def init_net(self, metadata):
        """
        Return Deeplab-v3 if on initial learning step
        """
        if metadata['cur_step'] == 0:
            net = modules.Deeplabv3(self.args['num_classes'], self.args['distributed'])
        else:
            args = deepcopy(self.args)
            net = modules.transnet.TransNet(args, metadata, self.net_configs, self.method_configs)

        return net

    def _init_old_net(self, metadata):
        self.logger.info("Initializing old network...")
        args = deepcopy(self.args)
        args['distributed'] = False
        args['pretrain'] = False
        self.old_net = modules.transnet.TransNet(args, metadata, self.net_configs, self.method_configs)
        self.old_net.to(metadata['device'])
        utils.checkpoint.load_model(
            self.old_net,
            self.args['logging_path'] + "/best_old_model",
            metadata,
            strict=False
        )
        utils.checkpoint.load_model(
            self.old_net.projector,
            self.net_configs['CKPT_PATH'] + str(metadata['cur_step']),
            metadata,
            strict=True
        )
        models.utils.freeze_all(self.old_net)

    def _init_new_net(self, metadata):
        if self.args['distributed']:
            net = self.net.module
        else:
            net = self.net
        if not isinstance(net, modules.transnet.TransNet):
            self.logger.info("Initializing current network...")
            self.net = None
            args = deepcopy(self.args)
            args['pretrain'] = False
            net = modules.transnet.TransNet(args, metadata, self.net_configs, self.method_configs)
            utils.checkpoint.load_model(
                net, self.args['logging_path'] + "/best_old_model", metadata, strict=False
            )
            net = DistributedDataParallel(net, device_ids=[metadata['device']])
            self.net = net

        self.logger.info("Loading projector for new_net")
        utils.checkpoint.load_model(
            net.projector,
            self.net_configs['CKPT_PATH'] + str(metadata['cur_step']),
            metadata,
            strict=True
        )

    def prepare_model_for_new_task(self, args, dist_args, metadata, datasets) -> None:
        self.cur_step = metadata['cur_step']
        self._compute_offsets(self.cur_step)
        if self.method_configs['label_type'] == "kd":
            self.loss_ce = utils.loss.UnbiasedCELoss(self.num_old_cls)
        self.loss_fs = utils.loss.FeatureSeperationLoss(
            factor=self.method_configs['weight_fs'],
            margin=self.method_configs['fs_margin'],
            num_class=args['num_classes'],
            num_old_class=self.num_old_cls,
            no_bkg=args['overlap']
        )

        if self.cur_step > 0:
            self.old_net = None
            if not os.path.exists(self.net_configs['CKPT_PATH'] + str(self.cur_step)):
                self.logger.info("Start training TransNet...")
                trainer = modules.transnet.TransNetTrainer(
                    args, dist_args, metadata,
                    self.method_configs, self.net_configs, idx=metadata['cur_step']
                )
                trainer.train_transnet(datasets, metadata)
                del trainer
            self._init_old_net(metadata)
            self._init_new_net(metadata)
            self.__cal_entropy_thres(args, dist_args, metadata, datasets)


    def train(self, x: Tensor, y: Tensor, metadata) -> Tuple[Tensor, Dict[str, float]]:
        layer = self.method_configs['proto_layer']
        output = self.net(x)
        output['logits'] = output['logits'][:, :self.num_new_cls]

        if metadata['cur_epoch'] >= self.method_configs['starting_epoch'] \
                and (self.method_configs['weight_fc'] > 0. or self.method_configs['weight_fs'] > 0.):
            self.__update_protos(output[layer].detach(), y, self.prototypes, self.count_features)
            loss_fc = self.loss_fc(output[layer], y, self.prototypes)
            loss_fs = self.loss_fs(output[layer], None, y, self.prototypes)
        else:
            loss_fc = torch.tensor(0., requires_grad=False, device=metadata['device'])
            loss_fs = torch.tensor(0., requires_grad=False, device=metadata['device'])

        if self.cur_step > 0:
            with torch.no_grad():
                target = self.old_net(x)
            target['logits'] = target['logits'][:, :self.num_old_cls]

            if self.loss_kd is not None:
                loss_kd = self.loss_kd(
                    output['logits'],
                    target['logits']
                )
            else:
                loss_kd = torch.tensor(0., requires_grad=False, device=metadata['device'])

            loss_ft = self.loss_ft(output['trans_f'], target['trans_f'])

            loss_intra = self.loss_intra(
                output[layer], target[layer], target['logits'], y, self.prototypes, self.num_old_cls
            )
            loss_inter = self.loss_inter(
                output[layer], target[layer], target['logits'], y, self.prototypes, self.num_old_cls
            )

            if self.method_configs['label_type'] == "pseudo":
                y = self.__gen_pseudo(y, target['logits'])
            loss_ce = self.method_configs['weight_ce'] * self.loss_ce(output['logits'], y)
            del target

            loss = loss_ce + loss_kd + loss_fc + loss_fs + loss_intra + loss_inter + loss_ft
            loss_dict = {
                "loss": loss.item(),
                "l_ce": loss_ce.item(),
                "l_kd": loss_kd.item(),
                "l_fc": loss_fc.item(),
                "l_fs": loss_fs.item(),
                "l_intra": loss_intra.item(),
                "l_inter": loss_inter.item(),
                "l_ft": loss_ft.item()
            }
        else:
            loss_ce = self.method_configs['weight_ce'] * self.loss_ce(output['logits'], y)
            loss = loss_ce + loss_fc + loss_fs
            loss_dict = {
                "loss": loss.item(),
                "l_ce": loss_ce.item(),
                "l_fc": loss_fc.item(),
                "l_fs": loss_fs.item()
            }

        rt_preds = output['logits'].detach().max(dim=1)[1].cpu().numpy()
        del output, x, y

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return rt_preds, loss_dict


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            output = self.net(x)
        embeddings = output['embeddings']
        preds = self.__gen_prediction(output['logits'])
        rt_dict = {
            "logits": output['logits'].detach().cpu(),
            "preds": preds.cpu(),
            "embeddings": embeddings.detach().cpu()
        }
        return rt_dict


    def __gen_pseudo(self, label: Tensor, old_logits: Tensor):
        pseudo = old_logits.max(dim=1)[1]
        entropy = self.__entropy(torch.softmax(old_logits, dim=1))
        if self.method_configs['pseudo_per_class']:
            pseudo[entropy > self.thres[pseudo]] = 255
        else:
            pseudo[entropy > self.method_configs['pseudo_thres']] = 255
        gt_mask = (label == 0).long()
        pseudo = pseudo * gt_mask
        return label + pseudo

    def __gen_prediction(self, logits: Tensor):
        preds = logits.detach().max(dim=1)[1]
        if self.method_configs['filter_output']:
            mask_entropy = (
                self.__entropy(torch.softmax(logits, dim=1)) < self.method_configs['filter_thres']
            ).long()
            preds = preds * mask_entropy

        return preds

    def __entropy(self, probs: Tensor) -> Tensor:
        EPS = 1e-8
        C = probs.shape[1]

        return (-1 / math.log(C)) * torch.sum(probs * torch.log(probs + EPS), dim=1)


    def __update_protos(
        self, features: Tensor, labels: Tensor, 
        prototypes: Tensor, count_features: Tensor
    ):
        """
        It takes features from current net and GT to update prototypes for each
        class. Note that only classes with GT are updated
        """
        device = features.device
        b, c, h, w = features.shape
        labels = trans_F.resize(labels, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        cls_new = torch.unique(labels)

        if cls_new[0] == 0:
            cls_new = cls_new[1:]
        if cls_new.shape[0] == 0:
            return
        if cls_new[-1] == 255:
            cls_new = cls_new[:-1]

        features_cl_num = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        features_cl_sum = torch.zeros((self.num_classes, c), dtype=torch.float, device=device)
        features_cl_mean = torch.zeros((self.num_classes, c), dtype=torch.float, device=device)
        features = features.permute(0, 2, 3, 1).reshape(-1, c)
        labels = labels.permute(0, 2, 3, 1).reshape(-1, 1).squeeze()
        for cl in cls_new:
            assert cl >= self.num_old_cls and cl < self.num_new_cls
            features_cl = features[(labels == cl), :]
            features_cl_num[cl] = features_cl.shape[0]
            features_cl_sum[cl] = torch.sum(features_cl, dim=0)
            features_cl_mean[cl] = torch.mean(features_cl, dim=0)

        if self.args['distributed']:
            dist.all_reduce(features_cl_num, op=dist.ReduceOp.SUM)
            dist.all_reduce(features_cl_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(features_cl_mean, op=dist.ReduceOp.SUM)

        for cl in range(self.num_old_cls, self.num_new_cls):
            if features_cl_num[cl] <= 0:
                continue
            proto_running_mean = (features_cl_sum[cl] \
                + count_features[cl] * prototypes[cl]) \
                / (count_features[cl] + features_cl_num[cl])
            count_features[cl] += features_cl_num[cl]
            prototypes[cl] = proto_running_mean


    def __cal_entropy_thres(self, args, dist_args, metadata, datasets):
        """
        Find the distribution of entropy of each class
        """
        dataset = datasets['train']
        rank = dist_args["rank"] if args['distributed'] else 0
        num_bins = 500
        histogram = torch.zeros(
            self.num_old_cls, num_bins,
            dtype=torch.long,
            device=metadata['device']
        )

        if args['distributed']:
            _sampler = DistributedSampler(
                dataset, dist_args['world_size'],
                dist_args["rank"], drop_last=True, shuffle=False
            )
        else:
            _sampler = None

        _loader = DataLoader(
            dataset=dataset,
            batch_size=args['val_batch_size'],
            shuffle=False,
            sampler=_sampler,
            num_workers=args['num_workers'],
            pin_memory=True,
            drop_last=False
        )

        self.logger.info("Computing entropy distribution for each class")
        progress_bar = utils.logger.R0Tqdm(rank, len(_loader))
        for images, labels in _loader:
            images = images.to(metadata['device'], dtype=torch.float32)
            labels = labels.to(metadata['device'], dtype=torch.long)
            logits_old = self.old_net(images)['logits'][:, :self.num_old_cls]
            probs = torch.softmax(logits_old, dim=1)
            max_prob, pseudo_label = probs.max(dim=1)
            entropy = self.__entropy(probs)

            x = pseudo_label.long().view(-1)
            y = torch.clamp((entropy * num_bins).long(), max=num_bins - 1).view(-1)
            histogram.index_put_(
                (x, y),
                values = torch.ones_like(x, dtype=torch.long, device=metadata['device']),
                accumulate=True
            )
            progress_bar.update()
        progress_bar.close()

        if args['distributed']:
            dist.reduce(histogram, dst=0, op=dist.ReduceOp.SUM)

        __thres = []
        thres = torch.zeros(self.num_old_cls, dtype=torch.float, device=metadata['device'])
        if rank == 0:
            for cl in range(self.num_old_cls):
                total = torch.sum(histogram[cl])
                running_sum = 0
                for bin_idx, bin in enumerate(histogram[cl]):
                    running_sum += bin
                    if running_sum > total * self.method_configs['pseudo_percent']:
                        break

                _t = max(
                    bin_idx / num_bins, self.method_configs['pseudo_entropy_min']
                )
                thres[cl] = _t
                __thres.append(_t)
                if thres[cl] > self.method_configs['pseudo_entropy_max']:
                    thres[cl] = self.method_configs['pseudo_entropy_max']

        if args['distributed']:
            dist.broadcast(thres, src=0)
        _thres = []
        for i in range(len(thres)):
            _thres.append(round(thres[i].item(), 5))
        self.logger.info(f"deployed thres value for each class is {_thres}")
        for i in range(len(__thres)):
            __thres[i] = round(__thres[i], 5)
        self.logger.info(f"original thres value for each class is {__thres}")
        

        self.thres = thres
