import os
from shutil import copyfile
from typing import Dict, Any, Union

import torch
import torch.nn as nn

from models.base import BaseMethod
from .logger import R0Logger
from utils import logger


def save_ckpt(
        path: str,
        args: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        model_state_dict: Union[Dict, Dict[str, Dict]] = None,
        method_vars: Dict[str, Any] = None
    ) -> None:
    checkpoint = {
        "args": args,
        "metadata": metadata,
        "model_state_dict": model_state_dict,
        "method_vars": method_vars
    }

    torch.save(checkpoint, path)
    

def load_ckpt(
        args: Dict[str, Any] = None,
        dist_args: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Dict]:
    checkpoint = None
    rank = dist_args['rank'] if args['distributed'] else 0
    logger = R0Logger(__name__, rank)
    if args['ckpt'] is not None:
        if os.path.isfile(args['ckpt']):
            logger.info(f"Loading checkpoint {args['ckpt']}")
            checkpoint = torch.load(args['ckpt'], map_location=metadata['device'])
            logger.info(f"Loaded the checkpoint successfully")
        else:
            logger.info("No ckpt discovered, training from scratch")
    else:
        logger.info("No ckpt specified, training from scratch")

    return checkpoint


def copy_old_net_ckpt(
    args: Dict[str, Any] = None,
    dist_args: Dict[str, Any] = None,
) -> None:
    rank = dist_args['rank'] if args['distributed'] else 0
    logger = R0Logger(__name__, rank)
    if (args['ckpt'] is not None) and (not args['test_only']) and (rank == 0):
        if os.path.isfile(args['ckpt']):
            dir = os.path.dirname(os.path.abspath(args['ckpt']))
            if dir == args['logging_path']:
                return
            file_src = args['ckpt']
            file_dst = os.path.join(args['logging_path'], 'best_old_model')
            if os.path.isfile(file_dst):
                return
            os.makedirs(args['logging_path'], exist_ok=True)
            logger.info("Copying old_model ckpt...")
            try:
                copy_ckpt(file_src, file_dst)
            except:
                logger.info("old model ckpt copy failed")


def save_model(model: nn.Module, path: str) -> None:
    if hasattr(model, "module"):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    torch.save(model_state_dict, path)
    del model_state_dict


def load_model(model: nn.Module, path: str, metadata: Dict[str, Any], strict: bool=True) -> None:
    model_state_dict = torch.load(path, map_location=metadata['device'])
    if "model_state_dict" in model_state_dict:
        model_state_dict = model_state_dict['model_state_dict']['net_state_dict']

    if hasattr(model, "module"):
        model.module.load_state_dict(model_state_dict, strict=strict)
    else:
        model.load_state_dict(model_state_dict, strict=strict)
    del model_state_dict


def copy_ckpt(src: str, dest: str):
    copyfile(src, dest)


def filter_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    rt_dict = {}
    for key in metadata['saveable_keys']:
        rt_dict[key] = metadata[key]
    
    return rt_dict


def load_args(args, checkpoint):
    if args['ckpt_model_only']:
        pass
    else:
        for k in checkpoint['args'].keys():
            if k in args['none_saveable_keys']:
                continue
            args[k] = checkpoint['args'][k]


def load_metadata(metadata, checkpoint):
    if 'metadata' not in checkpoint:
        metadata['newstep_flag'] = True
    else:
        for k in metadata['saveable_keys']:
            metadata[k] = checkpoint['metadata'][k]

