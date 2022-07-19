import os
import time
from typing import Any, List, Dict, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from matplotlib.figure import Figure

from models.base import BaseMethod
from datasets.base import BaseDataset
import utils


def train_step(
    model: BaseMethod,
    datasets: Dict[str, BaseDataset],
    args: Dict[str, Any] = None,
    dist_args: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None,
) -> None:

    rank = dist_args["rank"] if args['distributed'] else 0
    
    if args['distributed']:
        train_sampler = DistributedSampler(
            datasets['train'], dist_args['world_size'], rank,
            shuffle=True, seed=args['seed'], drop_last=True
        )
        val_sampler = DistributedSampler(
            datasets['val'], dist_args['world_size'], rank,
            shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        dataset=datasets['train'],
        batch_size=args['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        dataset=datasets['val'],
        batch_size=args['val_batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=args['num_workers'],
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    if not metadata['ckpt_flag'] or metadata['newstep_flag']:
        metadata['train_iters'] = args['epochs'] * len(train_loader)
        model.init_optimizer_and_scheduler(args, metadata)
    metadata['ckpt_flag'] = False
    metadata['newstep_flag'] = False
 
    starting_epoch = metadata['cur_epoch']
    for epoch in range(starting_epoch, args['epochs']):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        print(f"Starting training of epoch {metadata['cur_epoch'] + 1} / {args['epochs']}")
        start_time = time.time()
        train_epoch(model, train_loader, args, dist_args, metadata)
        end_time = time.time()
        print(f"Time taken for training on epoch {metadata['cur_epoch'] + 1} "
            f" : {round((end_time - start_time) / 60, 2)} mins")
        metadata['cur_epoch'] = epoch + 1

        if rank == 0 and (metadata['cur_epoch'] < args['epochs']):
            print("Saving checkpoint...")
            utils.checkpoint.save_ckpt(
                args['latest_ckpt_path'],
                args, utils.checkpoint.filter_metadata(metadata),
                model.method_state_dict(),
                model.method_variables()
            )
        
        print("Start Validating...")
        valid_metrics = evaluate(model, val_loader, args, dist_args, metadata)
        utils.metrics.cal_step_miou(args, valid_metrics)

        if (args['use_best_model'] 
            and (valid_metrics[args['best_model_metric']] > metadata['best_mIoU'])
            and rank == 0
        ):
            print("Saving best model...")
            metadata['best_mIoU'] = valid_metrics[args['best_model_metric']]
            utils.checkpoint.save_model(model.net, args['best_model_path'])

        print(valid_metrics)
        if rank == 0:
            metadata['writer'].log_valid(metadata, valid_metrics)

        if args['distributed']:
            dist.barrier()

    if args['use_best_model'] and os.path.exists(args['best_model_path']):
        print(f"Restore to this step's best model...")
        utils.checkpoint.load_model(model.net, args['best_model_path'], metadata)

    utils.reset_metadata(metadata)
    metadata['cur_step'] += 1
    metadata['newstep_flag'] = True
    if rank == 0:
        print("Saving checkpoint of this step...")
        utils.checkpoint.save_ckpt(
            args['latest_ckpt_path'],
            args, utils.checkpoint.filter_metadata(metadata),
            {
                "net_state_dict":
                model.net.module.state_dict() if args['distributed'] else model.net.state_dict()
            },
            model.method_variables()
        )
    metadata['cur_step'] -= 1

    if args['distributed']:
        dist.barrier()


def train_epoch(
    model: BaseMethod,
    train_loader: DataLoader,
    args: Dict[str, Any] = None,
    dist_args: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
) -> float:

    avg_loss = dict()
    rank = dist_args["rank"] if args['distributed'] else 0
    metric = utils.metrics.SimSegMetrics(args['num_classes'], args['reduce_bkg'])
    model.net.train()
    iter_counter = 0

    for (images, labels) in train_loader:
        metadata['cur_iter'] += 1
        np_labels = labels.numpy()
        images = images.to(metadata['device'], dtype=torch.float32)
        labels = labels.to(metadata['device'], dtype=torch.long)

        preds, loss = model.train(images, labels, metadata)
        cur_lr = model.step_scheduler(loss['loss'])

        if rank == 0:
            update_loss(avg_loss, loss)
            if args['print_metrics']:
                metric.update(np_labels, preds)

        iter_counter += 1
        if (metadata['cur_iter'] % args['print_interval'] == 0) and rank == 0:
            average_loss(avg_loss, iter_counter)
            iter_counter = 0
            if args['print_metrics']:
                train_metrics = metric.get_results()
                train_metrics['train_loss'] = avg_loss
                metric.reset()
            else:
                train_metrics = {'train_loss': avg_loss}

            metadata['writer'].log_train(metadata, train_metrics)
            metadata['writer'].log_lr(metadata, cur_lr)
            reset_loss(avg_loss)

        if args['distributed']:
            dist.barrier()


def evaluate(
    model: BaseMethod,
    valid_loader: DataLoader,
    args: Dict[str, Any] = None,
    dist_args: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
) -> Dict[str, float]:
    
    metric = utils.metrics.SimSegMetrics(args['num_classes'], args['reduce_bkg'])
    model.net.eval()

    for (images, labels) in valid_loader:
        np_labels = labels.numpy()
        images = images.to(metadata['device'], dtype=torch.float32)
        output = model.forward(images)
        metric.update(np_labels, output['preds'].numpy())

    if args['distributed']:
        metric.all_reduce(metadata)

    return metric.get_results()


def test(
    model: BaseMethod,
    datasets: Dict[str, BaseDataset],
    args: Dict[str, Any] = None,
    dist_args: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
) -> Tuple[Dict[str, float], Figure]:

    rank = dist_args["rank"] if args['distributed'] else 0
    if args['distributed']:
        test_sampler = DistributedSampler(
            datasets['test'], dist_args['world_size'], rank,
            shuffle=False, drop_last=False)
    else:
        test_sampler = None

    test_loader = DataLoader(
        dataset=datasets['test'],
        batch_size=args['val_batch_size'],
        shuffle=False,
        sampler=test_sampler,
        num_workers=args['num_workers'],
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    test_metrics, confusion_matrix = evaluate(model, test_loader, args, dist_args, metadata)
    return test_metrics, confusion_matrix


def update_loss(avg_loss: Dict[str, float], loss: Dict[str, float]) -> None:
    if len(avg_loss) == 0:
        for k, v in loss.items():
            avg_loss[k] = v
    else:
        for k, v in loss.items():
            avg_loss[k] += v


def average_loss(avg_loss: Dict[str, float], num_iters: int) -> None:
    for k in avg_loss.keys():
        avg_loss[k] /= num_iters


def reset_loss(avg_loss) -> None:
    avg_loss.clear()