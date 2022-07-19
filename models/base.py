import os
import json
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Tuple, Any

from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter

import models
import utils


class BaseMethod(ABC):
    
    def __init__(self, args, dist_args, metadata) -> None:
        super().__init__()
        self.num_classes = args['num_classes']
        self.n_class_per_step = args['n_class_per_step']
        self.args = args
        self.dist_args = dist_args
        self.rank = dist_args['rank'] if dist_args else 0

        self.method_configs = {}
        self.net_configs = {}
        self.optimizer, self.scheduler = None, None
        self.method_vars = []

    @abstractmethod
    def init_net(self, args):
        """
        Define the network to be used in this method
        """
        pass

    def load_configs(self, args, metadata) -> None:
        if args['external_args'] is not None:
            if os.path.isfile(args['external_args']):
                f = open(args['external_args'])
                configs = json.load(f)
                for k, v in configs['method_configs'].items():
                    self.method_configs[k] = v
                for k, v in configs['net_configs'].items():
                    self.net_configs[k] = v
                self.logger.info(f"loaded method configs from {args['external_args']}")

        # log method configs to tensorboard
        if metadata['writer'] is not None and \
                (not metadata['ckpt_flag'] or metadata['newstep_flag']):
            metadata['writer'].log_method_configs(self.method_configs)
            metadata['writer'].log_net_configs(self.net_configs)
        

    def method_state_dict(self) -> Dict[str, Dict]:
        state_dicts = {}
        if self.args['distributed']:
            state_dicts['net_state_dict'] = self.net.module.state_dict()
        else:
            state_dicts['net_state_dict'] = self.net.state_dict()
        state_dicts['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dicts['scheduler_state_dict'] = self.scheduler.state_dict()
        return state_dicts

    def _method_state_dict(self) -> Dict[str, Dict]:
        pass

    def method_variables(self) -> Dict[str, Any]:
        method_vars = dict()
        for k in self.method_vars:
            method_vars[k] = self.__dict__[k]
        return method_vars

    def load_method_variables(self, method_vars: Dict[str, Any]) -> None:
        for k, v in method_vars.items():
            self.__dict__[k] = v

    def prepare_model_for_new_task(self, args, dist_args, metadata, datasets) -> None:
        pass

    def end_train_for_new_task(self, args, dist_args, metadata, datasets) -> None:
        pass

    def prepare_model_for_testing(self, args, dist_args, metadata, datasets) -> None:
        pass

    def init_optimizer_and_scheduler(self, args, metadata) -> None:
        self.logger.info(f"Initializing scheduler and optimizer")
        if args['distributed']:
            net = self.net.module
        else:
            net = self.net
        self.optimizer, self.scheduler = models.utils.get_optimizer(
            net.parameters(), args, metadata
        )

    def step_scheduler(self, val_metric: Optional[float] = None) -> None:
        """
        Take a step with the scheduler (should be called after each epoch)
        """
        if self.args['scheduler_type'] == 'Plateau':
            assert val_metric is not None
            self.scheduler.step(val_metric)
        else:
            self.scheduler.step()

        return self._get_lr()[0]

    def _get_lr(self) -> List[float]:
        lr = [group['lr'] for group in self.optimizer.param_groups]
        return lr

    def _compute_offsets(self, cur_step) -> Tuple[int, int]:
        """
        for mask out classifier during training
        """
        self.num_old_cls = sum(self.n_class_per_step[i] for i in range(0, cur_step))
        self.num_new_cls = sum(self.n_class_per_step[i] for i in range(0, cur_step+1))
        self.logger.info(f"Number of old classes: {self.num_old_cls}")
        self.logger.info(f"Number of new classes: {self.num_new_cls - self.num_old_cls}")

    def _preprocess_target(self, targets: Tensor, labels: Tensor):
        pass

    def _init_old_net(self, metadata):
        """
        Init self.old_net before each step starts, and load best checkpoint
        from previous step
        """
        self.logger.info("Initializing old network...")
        args = deepcopy(self.args)
        args['distributed'] = False
        args['pretrain'] = False
        if self.old_net is None:
            self.old_net = self.init_net(args)
            self.old_net.to(metadata['device'])
        utils.checkpoint.load_model(self.old_net, args['best_old_model_path'], metadata)
        models.utils.freeze_all(self.old_net)

    @abstractmethod
    def train(self, x: Tensor, y: Tensor, metadata) -> Tuple[Tensor, float]:
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        pass
