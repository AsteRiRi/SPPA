import os
from typing import List, Optional, Tuple, Dict, Any, Callable
from copy import deepcopy

import numpy as np
import torch
import torch.utils.data as data
import torch.distributed as dist
import PIL.Image as Image

from .tasks import tasks_voc, tasks_ade

class BaseDataset(data.Dataset):

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
        self.task = None
        self.train = train
        self.overlap = overlap
        self.transform = transform
        self.masking = masking
        self.masking_type = masking_type
        self.image_set = 'train' if train else 'val'
        self.reduce_bkg = reduce_bkg
        self.images = []

    def choose_task(
        self, args: Dict[str, Any], metadata: Dict[str, Any],
        step_l: int=None, step_h: Optional[int]=None,
        filter_image: bool=True
    ):
        """
        This function select images contain label(s) in given step(s),
            and mask out some labels based on given step(s)
        """
        assert step_l in self.task.keys()
        if step_h is not None:
            assert step_h in self.task.keys()
            assert step_h >= step_l
            
            if step_h == step_l:
                step_h = None

        self.labels_old = []
        self.labels = []
        
        for i in range(0, step_l):
            self.labels_old += self.task[i]

        if step_h is not None:
            for i in range(step_l, step_h+1):
                self.labels += self.task[i]
        else:
            self.labels += self.task[step_l]
        self.labels += [0]

        self.labels = sorted(list(set(self.labels)))
        self.labels_old = sorted(list(set(self.labels_old)))
        self.labels_cum = sorted(list(set(self.labels_old + self.labels)))

        if filter_image:
            if (not args['distributed']) or (args['distributed'] and dist.get_rank() == 0):
                idxs_path = self.root + f"/idxs/{self.image_set}"
                if self.overlap:
                    idxs_fname = f"{args['task']}_{step_l}-{step_h}_ov.npy"
                else:
                    idxs_fname = f"{args['task']}_{step_l}-{step_h}.npy"
                idxs_fpath = os.path.join(idxs_path, idxs_fname)
                if os.path.exists(idxs_fpath):
                    self.cur_data_idxs = np.load(idxs_fpath).tolist()
                else:
                    os.makedirs(idxs_path, exist_ok=True)
                    self.cur_data_idxs = self._filter_images(self.overlap)
                    np.save(idxs_fpath, np.array(self.cur_data_idxs, dtype=int))

                if args['distributed']:
                    size = torch.tensor(len(self.cur_data_idxs), device=metadata['device'] ,dtype=torch.int)
                    idxs_to_send = torch.tensor(self.cur_data_idxs, device=metadata['device'], dtype=torch.int)
                    dist.broadcast(size, src=0)
                    dist.broadcast(idxs_to_send, src=0)

            else:
                size = torch.zeros(1, device=metadata['device'], dtype=torch.int)
                dist.broadcast(size, src=0)
                idxs_to_recieve = torch.zeros(size.item(), device=metadata['device'], dtype=torch.int)
                dist.broadcast(idxs_to_recieve, src=0)
                self.cur_data_idxs = idxs_to_recieve.cpu().tolist()

        else:
            self.cur_data_idxs = [i for i in range(len(self.images))]

        if self.train:
            self.masking_value = 0
        else:
            self.masking_value = 255

        if self.masking:
            self.labels_to_keep = self.labels + [255]
        else:
            self.labels_to_keep = None

    def disable_transform(self):
        self.transform = None

    def choose_full_data(self):
        """
        After calling this, this dataset is the same as full voc dataset without
        masking. Calling choose_task again to cancel the effect
        """
        self.cur_data_idxs = [i for i in range(len(self.images))]
        self.labels_to_keep = None

    def disable_masking(self):
        """
        After calling this, will diable masking labels of dataset
        """
        self.labels_to_keep = None

    def __getitem__(self, index) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple(image, target):
                type is torch Tensor if using transform
                type is PIL Image if not using transform

        """
        image = Image.open(self.images[self.cur_data_idxs[index]][0]).convert('RGB')
        target = Image.open(self.images[self.cur_data_idxs[index]][1])
        
        if self.labels_to_keep is not None:
            target = np.array(target)
            mask = np.isin(target, self.labels_to_keep, invert=True)
            target[mask] = self.masking_value
            target = Image.fromarray(target)

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.cur_data_idxs)

    def _filter_images(self, overlap: bool=True) -> List[int]:
        """
        overlap=True: Find images contains any of the classes in self.labels
        overlap=False: Additionally discard img containing classes not in labels_cum
        """
        idxs = []
        labels = deepcopy(self.labels)
        labels.remove(0)
        labels_cum = self.labels_cum + [255]

        if overlap: 
            fil = lambda c: any(x in labels for x in c)
        else:
            fil = lambda c: any(x in labels for x in c) and all(x in labels_cum for x in c)

        for i in range(len(self.images)):
            target = Image.open(self.images[i][1])
            classes = np.unique(np.array(target))
            if fil(classes):
                idxs.append(i)
        return idxs
