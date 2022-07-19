import os
import time
import platform
import random
import logging
from typing import List, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel 

import utils
import datasets as dsts
import models
import train



logging.basicConfig(
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.WARNING
)

def main():
    args, dist_args= utils.argparser.get_args()

    if dist_args is not None:
        mp.spawn(main_worker, nprocs=dist_args["ngpus_per_node"], args=(args, dist_args))
    else:
        main_worker(None, args, None)


def main_worker(gpu, args: Dict[str, Any], dist_args: Dict[str, Any] = None):
    metadata = utils.init_metadata(args)

    if args['distributed']:
        dist_args['gpu'] = gpu
        dist_args['rank'] = dist_args['node_rank'] * dist_args['ngpus_per_node'] + gpu
        metadata['device'] = torch.device(f"cuda:{gpu}")
        rank = dist_args['rank']
        logger = utils.logger.R0Logger(__name__, rank)
        logger.info(f"Use {dist_args['world_size']} GPUs for distributed training")
        dist.init_process_group(
            backend="gloo" if platform.system() == "Windows" else 'nccl',
            init_method=dist_args['dist_url'],
            world_size=dist_args['world_size'],
            rank=dist_args['rank'])
    else:
        rank = 0
        metadata['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger = utils.logger.R0Logger(__name__, rank)
        logger.info("Distributed training is disabled")

    checkpoint = utils.checkpoint.load_ckpt(args, dist_args, metadata)
    if checkpoint is not None:
        metadata['ckpt_flag'] = True
        if not args['test_only']:
            utils.checkpoint.load_args(args, checkpoint)
            utils.checkpoint.load_metadata(metadata, checkpoint)

    utils.checkpoint.copy_old_net_ckpt(args, dist_args)

    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    torch.backends.cudnn.benchmark = True

    if rank == 0:
        os.makedirs(args['logging_path'], exist_ok=True)
        logger.info(f"all info will be logged to {args['logging_path']}")
        metadata['writer'] = utils.logger.TBLogger(args['logging_path'])
    else:
        metadata['writer'] = None

    logger.info("Preparing datasets...")
    datasets = dsts.get_dataset(args)

    logger.info("Preparing model...")
    model = models.get_models(args, dist_args, metadata)
    model.net.to(metadata['device'])

    if checkpoint is not None:
        try:
            model.net.load_state_dict(
                checkpoint['model_state_dict']['net_state_dict'],
                strict=args['ckpt_strict_load']
            )
        except Exception as e:
            logger.debug(e)
            model.net.load_state_dict(checkpoint, strict=args['ckpt_strict_load'])
        logger.info(f"Loaded network state dicts successfully")

    if args['distributed']:
        model.net = DistributedDataParallel(model.net, device_ids=[gpu])

    if checkpoint is not None:
        del checkpoint

        if not args['test_only']:
            if metadata['cur_step'] > args['steps'][-1]:
                logger.info(f"All training steps are finished")
                starting_step_idx = 0
            else:
                assert metadata['cur_step'] in args['steps']
                starting_step_idx = args['steps'].index(metadata['cur_step'])
                logger.info(f"Resuming from task {args['task']} "
                    f"step {metadata['cur_step']} epoch {metadata['cur_epoch'] + 1}")
        else:
            starting_step_idx = 0
    else:
        starting_step_idx = 0

    for cur_step in range(starting_step_idx, len(args['steps'])):
        if args['test_only']:
            logger.info("test_only is set, skip training")
            break
        metadata['cur_step'] = args['steps'][cur_step]
        if ((metadata['cur_step'] != args['steps'][starting_step_idx])  or (metadata['ckpt_flag'] == False)):
            utils.reset_metadata(metadata)

        logger.info(f"Preparing datasets for step {metadata['cur_step']}")
        datasets['train'].choose_task(args, metadata, metadata['cur_step'])
        if metadata['cur_step'] == args['steps'][-1]:
            datasets['val'].choose_full_data()
        else:
            datasets['val'].choose_task(args, metadata, 0, metadata['cur_step'])
        logger.info(f"Length of train set: {len(datasets['train'])}")
        logger.info(f"Length of val set: {len(datasets['val'])}")

        logger.info("Preparing for learning new step")
        model.prepare_model_for_new_task(args, dist_args, metadata, datasets)

        start_time = time.time()
        logger.info(f"Start training step {metadata['cur_step']} of task {args['task']}")
        train.train_step(model, datasets, args, dist_args, metadata)
        end_time = time.time()
        logger.info(f"Time taken for training on step {metadata['cur_step']} "
            f"of task {args['task']} : {round((end_time - start_time) / 60, 2)} mins")

        model.end_train_for_new_task(args, dist_args, metadata, datasets)

    logger.info("Test on all classes...")
    test_metrics = []
    datasets['test'].choose_full_data()
    model.prepare_model_for_testing(args, dist_args, metadata, datasets)
    test_metrics, confusion_matrix = train.test(model, datasets, args, dist_args, metadata)
    if rank == 0:
        utils.metrics.cal_step_miou(args, test_metrics)
        logger.log_test(test_metrics)
        metadata['writer'].log_test(metadata, test_metrics)
        metadata['writer'].writer.close()

    if not args['test_only']:
        logger.info("Saving model...")
        if rank == 0:
            if args['distributed']:
                torch.save(model.net.module.state_dict(), args['final_model_path'])
            else:
                torch.save(model.net.state_dict(), args['final_model_path'])
        logger.info(f"Final model is saved to {args['final_model_path']}")


if __name__ == "__main__":
    main()