import os
from typing import Callable

import torch.utils.data as data
import PIL.Image as Image

from .base import BaseDataset
from .tasks import tasks_voc

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}


class VOCSegmentationIncremental(BaseDataset):
    def __init__(
        self,
        root: str,
        train: bool=True,
        transform: Callable=None,
        task: str=None,
        overlap: bool=True,
        masking: bool=True,
        masking_type: str="current",
        reduce_bkg: bool=False
    ):

        self.root = root
        self.task = tasks_voc[task]
        self.train = train
        self.overlap = overlap
        self.transform = transform
        self.masking = masking
        self.masking_type = masking_type
        self.reduce_bkg = reduce_bkg
        self.image_set = 'train' if train else 'val'

        assert os.path.isdir(self.root), f'Dataset not found at location {self.root}'
        splits_dir = os.path.join(self.root, 'list')

        if self.train:
            mask_dir = os.path.join(self.root, 'SegmentationClassAug')
            assert os.path.exists(mask_dir), "SegmentationClassAug not found"
            split_f = os.path.join(splits_dir, 'train_aug.txt')
        else:
            split_f = os.path.join(splits_dir, 'val.txt')

        assert os.path.exists(split_f), f"split file not found"
            
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        self.images = [(
                os.path.join(self.root, x[0][1:]),
                os.path.join(self.root, x[1][1:])
            ) for x in file_names
        ]

        self.choose_full_data()
