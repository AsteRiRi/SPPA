from typing import Dict, Any
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

import datasets


class SimSegMetrics():
    def __init__(self, n_classes: int, ignore_bkg: bool=False) -> None:
        self.n_classes = n_classes
        self.ignore_bkg = ignore_bkg
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, labels_true:np.ndarray, labels_pred:np.ndarray) -> None:
        for lt, lp in zip(labels_true, labels_pred):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def _fast_hist(self, label_true, label_pred) -> np.ndarray:
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2
        )
        hist = hist.reshape(self.n_classes, self.n_classes)
        return hist

    def all_reduce(self, metadata: Dict[str, Any]) -> None:
        confusion_matrix = torch.tensor(self.confusion_matrix, device=metadata['device'])
        dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
        self.confusion_matrix = confusion_matrix.cpu().numpy()

    def get_hist(self, normalize_row: bool=True) -> np.array:
        """
        return confusion matrix
        """
        return deepcopy(self.confusion_matrix)

    def get_results(self) -> Dict[str, float]:

        EPS = 1e-6
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum >= 1e-4)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + EPS)
        acc_cls = np.mean(acc_cls_c[mask])
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)
        mean_iu = np.mean(iu[mask])
        return {
                "Overall_Acc": acc,
                "Mean_Acc": acc_cls,
                "Mean_IoU": mean_iu,
                "Class_IoU": iu,
            }

    def reset(self) -> None:
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def cal_step_miou(args, test_metrics):
    task = datasets.get_task(args)
    if args['dataset'] == "voc":
        old_cls_idx = task[0]
        new_cls_idx = [x for x in range(21)]
        for item in old_cls_idx:
            new_cls_idx.remove(item)
        test_metrics["old"] = np.mean(test_metrics['Class_IoU'][old_cls_idx])
        test_metrics["new"] = np.mean(test_metrics['Class_IoU'][new_cls_idx])
        test_metrics["all"] = np.mean(test_metrics['Class_IoU'])        

    elif args['dataset'] == "ade":
        old_cls_idx = task[0][1:]
        new_cls_idx = [x for x in range(1, 151)]
        for item in old_cls_idx:
            new_cls_idx.remove(item)
        test_metrics["old"] = np.mean(test_metrics['Class_IoU'][old_cls_idx])
        test_metrics["new"] = np.mean(test_metrics['Class_IoU'][new_cls_idx])
        test_metrics["all"] = np.mean(test_metrics['Class_IoU'])
