# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional

import mmcv
import torch
from mmcv import flowwrite
from mmcv.runner import get_dist_info
from mmflow.apis.test import collect_results_cpu, collect_results_gpu

Module = torch.nn.Module
DataLoader = torch.utils.data.DataLoader


def single_gpu_test(
        model: Module,
        data_loader: DataLoader,
        out_dir: Optional[str] = None):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(test_mode=True, **data)
        batch_size = len(result)

        for _ in range(batch_size):
            prog_bar.update()

        if out_dir is not None:
            mmcv.mkdir_or_exist(out_dir)
            for r, a in zip(result, data['img_metas'].data[0]):
                video_name, img_name = a['filename1'].split('/')[-2:]
                img_name = img_name.replace('img', 'flow')
                flowwrite(r['flow'], osp.join(out_dir, video_name, img_name), quantize=True)


def multi_gpu_test(
        model: Module,
        data_loader: DataLoader,
        out_dir: Optional[str] = None,
        tmpdir: Optional[str] = None,
        gpu_collect: bool = False):
    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(test_mode=True, **data)
            if result[0].get('flow', None) is not None:
                result = [_['flow'] for _ in result]
            elif result[0].get('flow_fw', None) is not None:
                result = [_['flow_fw'] for _ in result]

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

        if out_dir is not None:
            mmcv.mkdir_or_exist(out_dir)
            if gpu_collect:
                result = collect_results_gpu(result, len(dataset))
            else:
                result = collect_results_cpu(result, len(dataset), tmpdir)
                
            for r, a in zip(result, data['img_metas'].data[0]):
                video_name, img_name = a['filename1'].split('/')[-2:]
                img_name = img_name.replace('img', 'flow')
                flowwrite(r['flow'], osp.join(out_dir, video_name, img_name), quantize=True)
