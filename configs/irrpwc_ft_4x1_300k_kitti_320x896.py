img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False)
crop_size = (320, 896)
global_transform = dict(
    translates=(0.02, 0.02),
    zoom=(0.98, 1.02),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))
relative_transform = dict(
    translates=(0.0025, 0.0025),
    zoom=(0.99, 1.01),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))
sparse_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(
        type='ColorJitter',
        brightness=0.05,
        contrast=0.2,
        saturation=0.25,
        hue=0.1),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='GaussianNoise', sigma_range=(0, 0.04), clamp_range=(0.0, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        global_transform=dict(
            translates=(0.02, 0.02),
            zoom=(0.98, 1.02),
            shear=(1.0, 1.0),
            rotate=(-0.5, 0.5)),
        relative_transform=dict(
            translates=(0.0025, 0.0025),
            zoom=(0.99, 1.01),
            shear=(1.0, 1.0),
            rotate=(-0.5, 0.5)),
        check_bound=True),
    dict(type='RandomCrop', crop_size=(320, 896)),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=('img_fields', 'ann_fields', 'filename1', 'filename2',
                   'ori_filename1', 'ori_filename2', 'filename_flow',
                   'ori_filename_flow', 'ori_shape', 'img_shape',
                   'img_norm_cfg'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(type='InputResize', exponent=6),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=[
            'flow_gt', 'valid', 'filename1', 'filename2', 'ori_filename1',
            'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
            'scale_factor', 'pad_shape'
        ])
]
kitti2015_train = dict(
    type='KITTI2015',
    data_root='data/kitti2015',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', sparse=True),
        dict(
            type='ColorJitter',
            brightness=0.05,
            contrast=0.2,
            saturation=0.25,
            hue=0.1),
        dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(
            type='GaussianNoise',
            sigma_range=(0, 0.04),
            clamp_range=(0.0, 1.0)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.5, direction='vertical'),
        dict(
            type='RandomAffine',
            global_transform=dict(
                translates=(0.02, 0.02),
                zoom=(0.98, 1.02),
                shear=(1.0, 1.0),
                rotate=(-0.5, 0.5)),
            relative_transform=dict(
                translates=(0.0025, 0.0025),
                zoom=(0.99, 1.01),
                shear=(1.0, 1.0),
                rotate=(-0.5, 0.5)),
            check_bound=True),
        dict(type='RandomCrop', crop_size=(320, 896)),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs', 'flow_gt', 'valid'],
            meta_keys=('img_fields', 'ann_fields', 'filename1', 'filename2',
                       'ori_filename1', 'ori_filename2', 'filename_flow',
                       'ori_filename_flow', 'ori_shape', 'img_shape',
                       'img_norm_cfg'))
    ],
    test_mode=False)
kitti2015_val_test = dict(
    type='KITTI2015',
    data_root='data/kitti2015',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', sparse=True),
        dict(type='InputResize', exponent=6),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=[
                'flow_gt', 'valid', 'filename1', 'filename2', 'ori_filename1',
                'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                'scale_factor', 'pad_shape'
            ])
    ],
    test_mode=True)
kitti2012_train = ({
    'type':
    'KITTI2012',
    'data_root':
    'data/kitti2012',
    'pipeline': [{
        'type': 'LoadImageFromFile'
    }, {
        'type': 'LoadAnnotations',
        'sparse': True
    }, {
        'type': 'ColorJitter',
        'brightness': 0.05,
        'contrast': 0.2,
        'saturation': 0.25,
        'hue': 0.1
    }, {
        'type': 'RandomGamma',
        'gamma_range': (0.7, 1.5)
    }, {
        'type': 'Normalize',
        'mean': [0.0, 0.0, 0.0],
        'std': [255.0, 255.0, 255.0],
        'to_rgb': False
    }, {
        'type': 'GaussianNoise',
        'sigma_range': (0, 0.04),
        'clamp_range': (0.0, 1.0)
    }, {
        'type': 'RandomFlip',
        'prob': 0.5,
        'direction': 'horizontal'
    }, {
        'type': 'RandomFlip',
        'prob': 0.5,
        'direction': 'vertical'
    }, {
        'type': 'RandomAffine',
        'global_transform': {
            'translates': (0.02, 0.02),
            'zoom': (0.98, 1.02),
            'shear': (1.0, 1.0),
            'rotate': (-0.5, 0.5)
        },
        'relative_transform': {
            'translates': (0.0025, 0.0025),
            'zoom': (0.99, 1.01),
            'shear': (1.0, 1.0),
            'rotate': (-0.5, 0.5)
        },
        'check_bound': True
    }, {
        'type': 'RandomCrop',
        'crop_size': (320, 896)
    }, {
        'type': 'DefaultFormatBundle'
    }, {
        'type':
        'Collect',
        'keys': ['imgs', 'flow_gt', 'valid'],
        'meta_keys':
        ('img_fields', 'ann_fields', 'filename1', 'filename2', 'ori_filename1',
         'ori_filename2', 'filename_flow', 'ori_filename_flow', 'ori_shape',
         'img_shape', 'img_norm_cfg')
    }],
    'test_mode':
    False
}, )
kitti2012_val_test = dict(
    type='KITTI2012',
    data_root='data/kitti2012',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', sparse=True),
        dict(type='InputResize', exponent=6),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=[
                'flow_gt', 'valid', 'filename1', 'filename2', 'ori_filename1',
                'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                'scale_factor', 'pad_shape'
            ])
    ],
    test_mode=True)
data = dict(
    train_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=5,
        drop_last=True,
        shuffle=False,
        persistent_workers=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    train=[{
        'type':
        'KITTI2015',
        'data_root':
        'data/kitti2015',
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }, {
            'type': 'LoadAnnotations',
            'sparse': True
        }, {
            'type': 'ColorJitter',
            'brightness': 0.05,
            'contrast': 0.2,
            'saturation': 0.25,
            'hue': 0.1
        }, {
            'type': 'RandomGamma',
            'gamma_range': (0.7, 1.5)
        }, {
            'type': 'Normalize',
            'mean': [0.0, 0.0, 0.0],
            'std': [255.0, 255.0, 255.0],
            'to_rgb': False
        }, {
            'type': 'GaussianNoise',
            'sigma_range': (0, 0.04),
            'clamp_range': (0.0, 1.0)
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'horizontal'
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'vertical'
        }, {
            'type': 'RandomAffine',
            'global_transform': {
                'translates': (0.02, 0.02),
                'zoom': (0.98, 1.02),
                'shear': (1.0, 1.0),
                'rotate': (-0.5, 0.5)
            },
            'relative_transform': {
                'translates': (0.0025, 0.0025),
                'zoom': (0.99, 1.01),
                'shear': (1.0, 1.0),
                'rotate': (-0.5, 0.5)
            },
            'check_bound': True
        }, {
            'type': 'RandomCrop',
            'crop_size': (320, 896)
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type':
            'Collect',
            'keys': ['imgs', 'flow_gt', 'valid'],
            'meta_keys':
            ('img_fields', 'ann_fields', 'filename1', 'filename2',
             'ori_filename1', 'ori_filename2', 'filename_flow',
             'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg')
        }],
        'test_mode':
        False
    },
           ({
               'type':
               'KITTI2012',
               'data_root':
               'data/kitti2012',
               'pipeline': [{
                   'type': 'LoadImageFromFile'
               }, {
                   'type': 'LoadAnnotations',
                   'sparse': True
               }, {
                   'type': 'ColorJitter',
                   'brightness': 0.05,
                   'contrast': 0.2,
                   'saturation': 0.25,
                   'hue': 0.1
               }, {
                   'type': 'RandomGamma',
                   'gamma_range': (0.7, 1.5)
               }, {
                   'type': 'Normalize',
                   'mean': [0.0, 0.0, 0.0],
                   'std': [255.0, 255.0, 255.0],
                   'to_rgb': False
               }, {
                   'type': 'GaussianNoise',
                   'sigma_range': (0, 0.04),
                   'clamp_range': (0.0, 1.0)
               }, {
                   'type': 'RandomFlip',
                   'prob': 0.5,
                   'direction': 'horizontal'
               }, {
                   'type': 'RandomFlip',
                   'prob': 0.5,
                   'direction': 'vertical'
               }, {
                   'type': 'RandomAffine',
                   'global_transform': {
                       'translates': (0.02, 0.02),
                       'zoom': (0.98, 1.02),
                       'shear': (1.0, 1.0),
                       'rotate': (-0.5, 0.5)
                   },
                   'relative_transform': {
                       'translates': (0.0025, 0.0025),
                       'zoom': (0.99, 1.01),
                       'shear': (1.0, 1.0),
                       'rotate': (-0.5, 0.5)
                   },
                   'check_bound': True
               }, {
                   'type': 'RandomCrop',
                   'crop_size': (320, 896)
               }, {
                   'type': 'DefaultFormatBundle'
               }, {
                   'type':
                   'Collect',
                   'keys': ['imgs', 'flow_gt', 'valid'],
                   'meta_keys': ('img_fields', 'ann_fields', 'filename1',
                                 'filename2', 'ori_filename1', 'ori_filename2',
                                 'filename_flow', 'ori_filename_flow',
                                 'ori_shape', 'img_shape', 'img_norm_cfg')
               }],
               'test_mode':
               False
           }, )],
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='KITTI2015',
                data_root='data/kitti2015',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', sparse=True),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'valid', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True),
            dict(
                type='KITTI2012',
                data_root='data/kitti2012',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', sparse=True),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'valid', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True)
        ],
        separate_eval=True),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='KITTI2015',
                data_root='data/kitti2015',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', sparse=True),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'valid', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True),
            dict(
                type='KITTI2012',
                data_root='data/kitti2012',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', sparse=True),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'valid', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True)
        ],
        separate_eval=True))
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl', port=12345)
log_level = 'INFO'
load_from = 'work_dir/irrt/latest.pth'
resume_from = None
workflow = [('train', 1)]
optimizer = dict(
    type='Adam', lr=3e-05, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='MultiStage',
    by_epoch=False,
    gammas=[0.5, 0.5],
    milestone_lrs=[3e-05, 2e-05],
    milestone_iters=[0, 150000],
    steps=[[
        45000, 65000, 85000, 95000, 97500, 100000, 110000, 120000, 130000,
        140000
    ],
           [
               195000, 215000, 235000, 245000, 247500, 250000, 260000, 270000,
               280000, 290000
           ]])
runner = dict(type='IterBasedRunner', max_iters=300000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=50000, metric='EPE')
model = dict(
    type='IRRPWC',
    encoder=dict(
        type='PWCNetEncoder',
        in_channels=3,
        net_type='Small',
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ],
        out_channels=(16, 32, 64, 96, 128, 196),
        strides=(2, 2, 2, 2, 2, 2),
        dilations=(1, 1, 1, 1, 1, 1),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    decoder=dict(
        type='IRRPWCDecoder',
        flow_levels=[
            'level0', 'level1', 'level2', 'level3', 'level4', 'level5',
            'level6'
        ],
        corr_in_channels=dict(
            level2=32, level3=64, level4=96, level5=128, level6=196),
        corr_feat_channels=32,
        flow_decoder_in_channels=115,
        occ_decoder_in_channels=114,
        corr_cfg=dict(type='Correlation', max_displacement=4),
        scaled=True,
        warp_cfg=dict(type='Warp', align_corners=True),
        densefeat_channels=(128, 128, 96, 64, 32),
        flow_post_processor=dict(
            type='ContextNet',
            in_channels=565,
            out_channels=2,
            feat_channels=(128, 128, 128, 96, 64, 32),
            dilations=(1, 2, 4, 8, 16, 1),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        flow_refine=dict(
            type='FlowRefine',
            in_channels=35,
            feat_channels=(128, 128, 64, 64, 32, 32),
            patch_size=3,
            warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        occ_post_processor=dict(
            type='ContextNet',
            in_channels=563,
            out_channels=1,
            feat_channels=(128, 128, 128, 96, 64, 32),
            dilations=(1, 2, 4, 8, 16, 1),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        occ_refine=dict(
            type='OccRefine',
            in_channels=65,
            feat_channels=(128, 128, 64, 64, 32, 32),
            patch_size=3,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
            warp_cfg=dict(type='Warp', align_corners=True)),
        occ_upsample=dict(
            type='OccShuffleUpsample',
            in_channels=11,
            feat_channels=32,
            infeat_channels=16,
            out_channels=1,
            warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        occ_refined_levels=['level0', 'level1'],
        flow_div=20.0,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        flow_loss=dict(
            type='MultiLevelEPE',
            weights=dict(
                level6=0.32,
                level5=0.08,
                level4=0.02,
                level3=0.01,
                level2=0.005,
                level1=0.00125,
                level0=0.0003125),
            p=1,
            q=0.4,
            eps=0.01,
            resize_flow='upsample',
            reduction='sum')),
    init_cfg=dict(
        type='Kaiming',
        a=0.1,
        nonlinearity='leaky_relu',
        layer=['Conv2d', 'ConvTranspose2d'],
        mode='fan_in',
        bias=0),
    train_cfg=dict(),
    test_cfg=dict())
custom_hooks = [dict(type='EMAHook')]
find_unused_parameters = True
work_dir = 'work_dir/irrft_k_check'
gpu_ids = range(0, 1)