# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from glob import glob

from mmflow.datasets.base_dataset import BaseDataset
from mmflow.datasets import DATASETS
import copy
from pathlib import Path
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Union

import mmcv
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from mmflow.datasets.pipelines import Compose


@DATASETS.register_module()
class InferenceOnly(Dataset):

    def __init__(self,
                 data_root: str,
                 pipeline: Sequence[dict],
                 video_names=None,
                 filename_tmpl='img_*.jpg',
                 test_mode: bool = False) -> None:
        super().__init__()
        self.data_root = data_root
        if video_names is not None:
            with open(video_names) as file:
                video_names = [line.rstrip() for line in file.readlines()]
            self.video_names = video_names
        else:
            self.video_names = [p.name for p in Path(data_root).iterdir()]
        self.pipeline = Compose(pipeline)
        self.filename_tmpl = filename_tmpl
        self.test_mode = test_mode
        self.dataset_name = self.__class__.__name__
        self.data_infos = []
        self.video_len = []
        self.load_data_info()
        self.cum_len = np.cumsum(self.video_len)

    def load_data_info(self) -> None:
        for vn in self.video_names:
            images = sorted(glob(osp.join(self.data_root, vn, self.filename_tmpl)))
            for i in range(len(images) - 1):
                data_info = dict(
                    img_info=dict(
                        filename1=images[i], filename2=images[i + 1]),
                    ann_info=dict())
                self.data_infos.append(data_info)
            self.video_len.append(len(images) - 1)

    def pre_pipeline(self, results: dict) -> None:
        results['img_fields'] = ['img1', 'img2']
        # results['ann_fields'] = []
        # results['img1_dir'] = self.data_root
        # results['img2_dir'] = self.data_root

    def prepare_data(self, idx: int) -> dict:
        results = copy.deepcopy(self.data_infos[idx])
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> dict:
        return self.prepare_data(idx)

