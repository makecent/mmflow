# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmflow import digit_version
from mmflow.datasets import build_dataloader, build_dataset
from mmflow.models import build_flow_estimator
from mmflow.utils import setup_multi_processes

from gpu_test import single_gpu_test, multi_gpu_test


def parse_args():
    parser = argparse.ArgumentParser(
        description=('Test (and eval)'
                     ' a flow estimator'))
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--src-dir', help='directory where rgb frames stored')
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

    assert args.out_dir, \
        ('Please specify at least one operation (save/eval/show the '
         'results / save the results) with the argument "--out-dir", "--eval"'
         ', "--show" or "--show-dir"')
    if args.out_dir is not None:
        mmcv.mkdir_or_exist(args.out_dir)

    cfg = Config.fromfile(args.config)
    if args.src_dir is not None:
        cfg.data.test.data_root = args.src_dir
    if args.video_names is not None:
        cfg.data.test.video_names = args.video_names
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set multi-process settings
    setup_multi_processes(cfg)

    # The overall dataloader settings
    loader_cfg = {
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    }
    # The specific training dataloader settings
    test_loader_cfg = {**loader_cfg, **cfg.data.get('test_dataloader', {})}

    # build the dataloader
    separate_eval = cfg.data.test.get('separate_eval', False)
    if separate_eval:
        # multi-datasets will be built as themselves.
        dataset = [
            build_dataset(dataset) for dataset in cfg.data.test.datasets
        ]
    else:
        # multi-datasets will be concatenated as one dataset.
        dataset = [build_dataset(cfg.data.test)]
    data_loader = [
        build_dataloader(
            _dataset,
            **test_loader_cfg,
            dist=distributed,
        ) for _dataset in dataset
    ]

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

    for i, i_data_loader in enumerate(data_loader):
        if args.out_dir:

            if not distributed:
                single_gpu_test(model, i_data_loader, args.out_dir)
            else:
                multi_gpu_test(model, i_data_loader, args.outdir)


if __name__ == '__main__':
    main()
