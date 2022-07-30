# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import warnings
from glob import glob
from typing import List

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmflow import digit_version
from mmflow.apis import single_gpu_test, multi_gpu_test
from mmflow.datasets import DATASETS
from mmflow.datasets import build_dataloader, build_dataset
from mmflow.datasets.pipelines import Compose
from mmflow.models import build_flow_estimator
from mmflow.utils import setup_multi_processes
from torch.utils.data import Dataset


@DATASETS.register_module()
class InferenceOnly(Dataset):

    def __init__(self,
                 video_path: str,
                 total_frames: int = None,
                 video_fmt=False,
                 imgname_tmpl='*',
                 test_mode: bool = False,
                 pipeline: List = [dict(type='LoadImageFromFile'),
                                   dict(type='InputPad', exponent=3),
                                   dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5],
                                        to_rgb=False),
                                   dict(type='TestFormatBundle'),
                                   dict(type='Collect', keys=['imgs'],
                                        meta_keys=['filename1', 'filename2', 'ori_filename1', 'ori_filename2',
                                                   'ori_shape', 'img_shape', 'img_norm_cfg', 'scale_factor',
                                                   'pad_shape', 'pad'])]) -> None:
        super().__init__()
        self.video_path = video_path
        if video_fmt:
            from decord import VideoReader
            total_frames = VideoReader(video_path)._num_frame
            self.pipeline = Compose(pipeline[1:])
        else:
            self.pipeline = Compose(pipeline)
        self.total_frames = total_frames
        self.video_fmt = video_fmt

        self.imgname_tmpl = imgname_tmpl
        self.test_mode = test_mode
        self.dataset_name = self.__class__.__name__
        self.data_infos = []
        self.load_data_info()

    def load_data_info(self) -> None:
        if self.video_fmt:
            # build a fake data_info
            for i in range(self.total_frames - 1):
                data_info = dict()
                self.data_infos.append(data_info)
        else:
            images = sorted(glob(osp.join(self.video_path, self.imgname_tmpl)))
            assert len(images) == self.total_frames
            for i in range(len(images) - 1):
                data_info = dict(
                    img_info=dict(filename1=images[i], filename2=images[i + 1]),
                    ann_info=dict())
                self.data_infos.append(data_info)

    def prepare_data(self, idx: int) -> dict:
        if self.video_fmt:
            from decord import VideoReader
            # I had tried to create the video reader as an attribute of the dataset to avoid redundant video reading
            # But it has BUG with torch dataloader.
            img1, img2 = VideoReader(self.video_path).get_batch([idx, idx + 1]).asnumpy()
            results = dict(img1=img1, img2=img2)
            # Set initial values for default meta_keys
            results['filename1'] = osp.join(self.video_path, f'{idx}.jpg')
            results['filename2'] = osp.join(self.video_path, f'{idx + 1}.jpg')
            results['ori_filename1'] = f'{idx}.jpg'
            results['ori_filename2'] = f'{idx + 1}.jpg'
            results['img_shape'] = img1.shape
            results['ori_shape'] = img1.shape
            results['pad_shape'] = img1.shape
            results['scale_factor'] = np.array([1.0, 1.0])
            num_channels = 1 if len(img1.shape) < 3 else img1.shape[2]
            results['img_norm_cfg'] = dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False)
        else:
            results = copy.deepcopy(self.data_infos[idx])
        results['img_fields'] = ['img1', 'img2']
        return self.pipeline(results)

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> dict:
        return self.prepare_data(idx)


def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a flow estimator')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('src_dir', help='directory where videos/frames stored')
    parser.add_argument('--raw-video', action='store_true', help='Whether the format of source files are videos')
    parser.add_argument('--level', type=int, choices=[1, 2], default=1, help='directory level of data')
    parser.add_argument('--imgname_tmpl', type=str, help='only images with name meeting the template are visible')
    parser.add_argument('--flowname_tmpl', type=str, default='flow_{:05}.jpg', help='the name template of flow files')
    parser.add_argument('--video-names', help='list of video names to be processed in src_dir')
    parser.add_argument('--out-dir', help='directory to save the flow file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    if args.out_dir is None:
        warnings.warn("Output directory was not specified, so dry-run only")
    else:
        mmcv.mkdir_or_exist(args.out_dir)

    if args.video_names is None:
        if args.level == 2:
            classes = os.listdir(args.src_dir)
            for classname in classes:
                new_dir = osp.join(args.out_dir, classname)
                if not osp.isdir(new_dir):
                    print(f'Creating folder: {new_dir}')
                    os.makedirs(new_dir)
        video_paths = glob(args.src_dir + '/*' * args.level)
    else:
        video_paths = [osp.join(args.src_dir, n.strip().split(' ')[0].split(',')[0]) for n in
                       open(args.video_names).readlines()]

    cfg = Config.fromfile(args.config)
    cfg.data.test = mmcv.ConfigDict(type='InferenceOnly')
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.imgname_tmpl is not None:
        cfg.data.test.imgname_tmpl = args.imgname_tmpl
    if args.raw_video:
        cfg.data.test.video_fmt = True

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set multi-process settings
    setup_multi_processes(cfg)

    # build the model and load checkpoint
    model = build_flow_estimator(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
    rank, _ = get_dist_info()

    # build the dataloader
    for video_path in video_paths:
        cfg.data.test.video_path = video_path
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=distributed)

        if not distributed:
            results = single_gpu_test(model, data_loader)
        else:
            results = multi_gpu_test(model, data_loader)

        if rank == 0:
            if args.out_dir:
                video_path = video_path.rsplit('.')[0]
                out_video_path = osp.join(args.out_dir, osp.relpath(video_path, args.src_dir))
                mmcv.mkdir_or_exist(out_video_path)
                print(f'\nwriting results to {out_video_path}')
                for i, i_result in enumerate(results):
                    if not distributed:
                        # Fix a bug in single_gpu_test that it does NOT take 'flow' filed of result.
                        i_result = i_result['flow']
                    flow_name = args.flowname_tmpl.format(i)
                    mmcv.flowwrite(i_result, osp.join(out_video_path, flow_name), quantize=True)


if __name__ == '__main__':
    main()
