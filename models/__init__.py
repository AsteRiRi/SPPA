import importlib

from .base import BaseMethod
from . import utils

def get_models(args, dist_args, metadata) -> BaseMethod:
    try:
        model = importlib.import_module('models.' + args['method'])
    except Exception as e:
        print(e)
        raise NotImplementedError(f"model {args['method']} is not supported")
    else:
        method = model.Method(args, dist_args, metadata)
        return method
