from copy import deepcopy
import logging
import time
import inspect
from typing import Union, List, Dict, Any, Tuple
from numpy import array

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, path) -> None:
        self.writer = SummaryWriter(path)

    def log_train(self, metadata: Dict[str, Any], train_metrics: Dict[str, float]):
        """
        This function logs training metrics to tensorboard
        """
        self.writer.add_scalars(
            f"Step_{metadata['cur_step']}/Loss",
            train_metrics['train_loss'],
            metadata['cur_iter']
        )
        if 'Mean_IoU' in train_metrics:
            self.writer.add_scalar(
                f"Step_{metadata['cur_step']}/Mean_IoU",
                train_metrics['Mean_IoU'],
                metadata['cur_iter']
            )

        self.writer.flush()

    def log_valid(self, metadata: Dict[str, Any], valid_metrics: Dict[str, float]):
        self.writer.add_scalar(
            f"Step_{metadata['cur_step']}/Val_old_mIoU",
            valid_metrics['old'],
            metadata['cur_iter']
        )
        self.writer.add_scalar(
            f"Step_{metadata['cur_step']}/Val_new_mIoU",
            valid_metrics['new'],
            metadata['cur_iter']
        )
        self.writer.add_scalar(
            f"Step_{metadata['cur_step']}/Val_all_mIoU",
            valid_metrics['all'],
            metadata['cur_iter']
        )

        self.writer.flush()
    
    def log_step(self, metadata: Dict[str, Any], miou: float):
        self.writer.add_scalar(f"Step_{metadata['cur_step']}/Best_IoU", miou)
        self.writer.flush()

    def log_test(self, metadata: Dict[str, Any], test_metric: List[Dict[str, Any]]):
        self.writer.add_scalars(
            "Test_Results",
            {
                "Old_class": test_metric['old'],
                "New_class": test_metric['new'],
                "All_class": test_metric['all']
            }
        )

        self.writer.flush()

    def log_lr(self, metadata, lr: float):
        self.writer.add_scalar(
            f"Step_{metadata['cur_step']}/Learning Rate",
            lr,
            metadata['cur_iter']
        )
        self.writer.flush()



class R0Logger:
    def __init__(self, name: str, rank: int) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.rank = rank

    def error(self, msg: str) -> None:
        if self.rank == 0:
            self.logger.error(msg)

    def warning(self, msg: str) -> None:
        if self.rank == 0:
            self.logger.warning(msg)

    def info(self, msg: str) -> None:
        if self.rank == 0:
            self.logger.info(msg)

    def debug(self, msg: str) -> None:
        if self.rank == 0:
            self.logger.debug(msg)

    def log_train(self, train_metrics: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        if self.rank == 0:
            msg = Msg.msg_train(train_metrics, metadata)
            self.logger.info(msg)

    def log_valid(self, val_metrics: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        if self.rank == 0:
            msg = Msg.msg_valid(val_metrics, metadata)
            self.logger.info(msg)

    def log_test(self, test_metrics: Dict[str, Any]) -> None:
        if self.rank == 0:
            msg_1, msg_2 = Msg.msg_test(test_metrics)
            self.logger.info(msg_1)
            self.logger.info(msg_2)



class Msg:
    @staticmethod
    def msg_train(train_metrics: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        
        msg = f"epoch:{metadata['cur_epoch'] + 1}" \
            + f" iteration:{metadata['cur_iter']}"

        for k, v in train_metrics['train_loss'].items():
            msg = msg + f" {k}:{round(v, 5)}"

        if 'Mean_IoU' in train_metrics:
            msg = msg + f" Overall_Acc:{round(train_metrics['Overall_Acc'], 5)}" \
                + f" Mean_IoU:{round(train_metrics['Mean_IoU'], 5)}"
        
        return msg

    @staticmethod
    def msg_valid(val_metrics: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        best_miuo = metadata['best_mIoU']

        if isinstance(val_metrics['new'], list):
            for i, data in enumerate(val_metrics['new']):
                val_metrics['new'][i] = round(data, 5)
        else:
            val_metrics['new'] = round(val_metrics['new'], 5)

        msg = f"Overall_Acc:{round(val_metrics['Overall_Acc'], 5)}" \
                + f" Mean_Acc:{round(val_metrics['Mean_Acc'], 5)}" \
                + f" Old_mIoU:{round(val_metrics['old'], 5)}" \
                + f" New_mIoU:{val_metrics['new']}" \
                + f" All_mIoU:{round(val_metrics['all'], 5)}" \
                + f" Best_mIoU:{round(best_miuo, 5)}" \
        
        return msg

    @staticmethod
    def msg_test(test_metrics: Dict[str, Any]) -> Tuple[str, str]:
        test_metrics = deepcopy(test_metrics)

        if isinstance(test_metrics['new'], list):
            for i, data in enumerate(test_metrics['new']):
                test_metrics['new'][i] = round(data, 5)
        else:
            test_metrics['new'] = round(test_metrics['new'], 5)

        msg_1 = f"Old_mIoU:{round(test_metrics['old'], 5)}" \
                + f" New_mIoU:{test_metrics['new']}" \
                + f" All_mIoU:{round(test_metrics['all'], 5)}" \

        for i, data in enumerate(test_metrics['Class_IoU']):
            test_metrics['Class_IoU'][i] = round(data, 5)

        msg_2 = f"Per_Class_mIoU:{test_metrics['Class_IoU']}"

        return msg_1, msg_2
