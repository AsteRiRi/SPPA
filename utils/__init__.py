from typing import Dict, Any

from . import argparser
from . import checkpoint
from . import logger
from . import metrics
from . import loss


def reset_metadata(metadata: Dict[str, Any]):
    metadata['cur_epoch'] = 0
    metadata['cur_iter'] = 0
    metadata['best_mIoU'] = 0.0


def init_metadata(args: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict()
    metadata['saveable_keys'] = [
        'best_mIoU', 'cur_step', 'cur_epoch', 'cur_iter', 'train_iters', 'newstep_flag'
    ]
    metadata['none_saveable_keys'] = ['device', 'ckpt_flag', 'writer']
    metadata['ckpt_flag'] = False
    metadata['newstep_flag'] = False
    metadata['cur_step'] = args['steps'][0]
    reset_metadata(metadata)

    return metadata
