import os
from typing import Callable

import torch.utils.data as data
import PIL.Image as Image

from .base import BaseDataset
from .tasks import tasks_ade


classes = [
    "void", "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane",
    "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant",
    "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror",
    "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
    "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter",
    "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs",
    "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop",
    "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship",
    "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool",
    "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan",
    "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
    "glass", "clock", "flag"
]



class AdeSegmentationIncremental(BaseDataset):

    def __init__(
        self,
        root: str,
        train: bool=True,
        transform: Callable=None,
        task: str=None,
        overlap: bool=True,
        masking: bool=True,
        masking_type="current",
        reduce_bkg: bool=True
    ):

        self.root = root
        self.task = tasks_ade[task]
        self.train = train
        self.overlap = overlap
        self.transform = transform
        self.masking = masking
        self.masking_type = masking_type
        self.reduce_bkg = reduce_bkg
        self.image_set = 'train' if train else 'val'

        if train:
            split = 'training'
        else:
            split = 'validation'
        annotation_folder = os.path.join(root, 'annotations', split)
        image_folder = os.path.join(root, 'images', split)

        fnames = sorted(os.listdir(image_folder))
        self.images = [(
            os.path.join(image_folder, fname),
            os.path.join(annotation_folder, fname[:-3] + "png"))
            for fname in fnames
        ]

        self.choose_full_data()
