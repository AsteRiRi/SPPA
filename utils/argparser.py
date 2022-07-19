import argparse
import logging
import math
import os
import time
import json
import random
from typing import Any, Dict, List

import torch

import datasets


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()

    # External config file
    parser.add_argument("--external_args", type=str, default='./configs/local.json')

    # Method
    parser.add_argument("--method", type=str, default='joint')

    # Datset Options
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--dataset", type=str, default='voc')
    parser.add_argument("--task", type=str, default="15-5")
    parser.add_argument("--steps", type=List[int], default=None)
    parser.add_argument("--overlap", action='store_true', default=True)
    parser.add_argument("--masking", action='store_true', default=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--crop_val", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=1)

    # Train Options
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=List[float], default=[7e-3])
    parser.add_argument("--lr_power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer_type", type=str, default="momentum")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--best_model_metric", type=str, default='all')

    # Validation Options
    parser.add_argument("--test_only", action='store_true', default=False)

    # miscellaneous
    parser.add_argument("--logging_path", type=str, default=None)
    parser.add_argument("--logging_tag", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt_model_only", action="store_true", default=False)
    parser.add_argument("--ckpt_strict_load", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--print_metrics", action="store_true", default=True)

    # Distributed and Performance Options
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--node_rank', default=0, type=int)
    parser.add_argument('--gpu_id', type=str, default=None)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--dist_url', default="env://", type=str)

    args = vars(parser.parse_args())
    args, dist_args = __process_args(args)
    return args, dist_args

def __process_args(args: Dict[str, Any]) -> Dict[str, Any]:

    # Specify keys that should NOT be loaded when resuming from a checkpoint
    args['none_saveable_keys'] = [
        'external_args', 'test_only', 'distributed', 'node_rank', 'gpu_id',
        'dist_url', 'ckpt'
    ]

    if not args['crop_val']:
        args['val_batch_size'] = 1

    # Replace args with external args
    if args['external_args'] is not None:
        if os.path.isfile(args['external_args']):
            f = open(args['external_args'])
            external_args = json.load(f)['args']
            for key in external_args.keys():
                args[key] = external_args[key]
            logger.info(f"{args['external_args']} loaded")
        else:
            logger.info("external_args does not exists, skipped")

    # Setting up logging path
    # Using current time as part of logging path
    localtime = time.localtime(time.time())
    run_time = f"{localtime[0]}-{localtime[1]}-{localtime[2]}_{localtime[3]}.{localtime[4]}.{localtime[5]}"
    if args['logging_tag'] is not None:
        if args['run_id'] is not None:
            suffix = args['logging_tag'] + args['run_id']
        else:
            suffix = args['logging_tag'] + run_time
    else:
        suffix = run_time
    args['logging_path'] = os.path.join(args['logging_path_base'], suffix)

    # Setting up ckpt path
    args['latest_ckpt_path'] = os.path.join(args['logging_path'], "latest_ckpt")
    args['best_model_path'] = os.path.join(args['logging_path'], "best_cur_model")
    args['best_old_model_path'] = os.path.join(args['logging_path'], "best_old_model")
    args['final_model_path'] = os.path.join(args['logging_path'], "final_model")
    
    # Length of lr list must equal to the steps
    if not args['test_only']:
        assert len(args['lr']) == len(args['steps'])
    
    # Setting up total number of classes
    if args['dataset'] == 'voc':
        args['num_classes'] = 21
        args['n_class_per_step'] = [len(i[1]) for i in datasets.tasks_voc[args['task']].items()]
    elif args['dataset'] == 'ade':
        args['num_classes'] = 151
        args['n_class_per_step'] = [len(i[1]) for i in datasets.tasks_ade[args['task']].items()]
    else:
        raise NotImplementedError

    # Setting up gpu(s) to use
    if args['gpu_id'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']

    # Setting up distributed training args
    if args['distributed']:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(random.randint(30000, 65535))
        dist_args = {"num_nodes": args['num_nodes'], "node_rank": args['node_rank']}
        dist_args['ngpus_per_node'] = torch.cuda.device_count()
        dist_args['world_size'] = dist_args['ngpus_per_node'] * dist_args['num_nodes']
        args['batch_size'] = int(math.ceil(args['batch_size'] / dist_args['world_size']))
        args['val_batch_size'] = int(math.ceil(args['val_batch_size'] / dist_args['world_size']))
        dist_args['dist_url'] = args['dist_url']
    else:
        dist_args = None

    return args, dist_args
