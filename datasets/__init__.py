from typing import List, Dict, Any

import torch
import torchvision.transforms.functional as trans_F

from .voc import VOCSegmentationIncremental
from .ade import AdeSegmentationIncremental
from .base import BaseDataset
from . import transform
from .tasks import tasks_voc, tasks_ade


def get_dataset(args) -> Dict[str, BaseDataset]:

    train_transform, val_transform = get_transform(args)

    if args['dataset'] == 'voc':
        dataset = VOCSegmentationIncremental
    elif args['dataset'] == 'ade':
        dataset = AdeSegmentationIncremental
    else:
        raise NotImplementedError(f"dataset {args['dataset']} is not supported")

    train_dst = dataset(
        root=args['data_root'],
        train=True,
        transform=train_transform,
        task=args['task'],
        overlap=args['overlap'],
        masking=args['masking'],
        masking_type=args['masking_type'],
        reduce_bkg=args['reduce_bkg']
    )

    if args['cross_val']:
        train_len = int(0.8 * len(train_dst))
        val_len = len(train_dst) - train_len
        train_dst, val_dst = torch.utils.data.random_split(
            train_dst, [train_len, val_len],
            generator=torch.Generator().manual_seed(args['seed'])
        )
    else:
        val_dst = dataset(
            root=args['data_root'],
            train=False,
            transform=val_transform,
            task=args['task'],
            overlap=True,
            masking=args['masking'],
            masking_type=args['masking_type'],
            reduce_bkg=args['reduce_bkg']
        )
    
    test_dst = dataset(
        root=args['data_root'],
        train=args['val_on_trainset'],
        transform=val_transform,
        task=args['task'],
        overlap=True,
        masking=True,
        masking_type="current",
        reduce_bkg=args['reduce_bkg']
    )

    datasets = {
        'train': train_dst,
        'val': val_dst,
        'test': test_dst
    }
    return datasets


def get_transform(args):

    if args['colorjitter']:
        train_transform = transform.Compose(
            [
                transform.RandomResizedCrop(args['crop_size'], (0.5, 2.0)),
                transform.RandomHorizontalFlip(),
                transform.ColorJitter(0.15, 0.5, 0.5, 0.1),
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_transform = transform.Compose(
            [
                transform.RandomResizedCrop(args['crop_size'], (0.5, 2.0)),
                transform.RandomHorizontalFlip(),
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        
    if args['crop_val']:
        val_transform = transform.Compose(
            [
                transform.Resize(size=args['crop_size']),
                transform.CenterCrop(size=args['crop_size']),
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        val_transform = transform.Compose(
            [
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return train_transform, val_transform


def denormalize(image):
    image = trans_F.normalize(
        image,
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        inplace=True
    )
    return image


def get_task(args: Dict[str, Any]) -> Dict[str, Dict]:
    if args['dataset'] == 'voc':
        task = tasks_voc[args['task']]
    elif args['dataset'] == 'ade':
        task = tasks_ade[args['task']]
    else:
        raise NotImplementedError

    return task