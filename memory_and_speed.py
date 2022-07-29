import torch
from mmflow.apis import init_model

# Specify the path to model config and checkpoint file
config_file = 'configs/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.py'
checkpoint_file = 'checkpoints/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

from mmcv import track_iter_progress
for i in track_iter_progress(list(range(10000))):
    # loss = model(imgs=torch.rand(1, 6, 256, 256).cuda(),
    #              flow_gt=torch.rand(1, 2, 256, 256).cuda(),
    #              flow_init=torch.zeros((1, 2, 256, 256)).cuda(),
    #              valid=None,
    #              test_mode=False)
    # loss['loss_flow'].backward()

    with torch.no_grad():
        out = model(imgs=torch.rand(1, 6, 224, 224).cuda())


# %% GMA (1, 6, 224, 224)
# GPU memory 1085 MB
# 10.1 task/s (single 2080 ti)
# 6.5 kb per image (flow_x + flow_y)
# TH14 training (trimmed) 721024/10.1/3600 = 19.8 hours
# TH14 training (untrimmed) (363343+1220645)/10.1/3600 = 43.6 hours
# TH14 testing (untrimmed, sampled) 210000/10.1/3600 = 5.77 hours
# TH14 testing (untrimmed) 1344850/10.1/3600 = 37.0 hours

# TH14 training (untrimmed) (363343+1220645)*6.5/1000000 = 10.3 Gbytes
# TH14 testing (untrimmed) 1344850*6.5/1000000  = 8.7 Gbytes

# %% PWC (1, 6, 256, 256)
# GPU memory 1859 MB
# 93.4 task/s (single 2080 ti)
# 8.3 kb per image (flow_x + flow_y)
# TH14 training (trimmed) 721024/93.4/3600 = 2.14 hours
# TH14 training (untrimmed) (363343+1220645)/93.4/3600 = 4.71 hours
# TH14 testing (untrimmed, sampled) 210000/93.4/3600 = 0.62 hours
# TH14 testing (untrimmed) 1344850/93.4/3600 = 4.0 hours

# TH14 training (untrimmed) (363343+1220645)*8.3/1000000 = 13.1 Gbytes
# TH14 testing (untrimmed) 1344850*8.3/1000000  = 11.2 Gbytes

# %% FlowNet2 (1, 6, 256, 256)
# GPU memory 2143 MB
# 77.0 task/s (single 2080 ti)

# %% LiteFlowNet (1, 6, 256, 256)
# GPU memory 1325 MB
# 68.4 task/s (single 2080 ti)

# %% IRR (1, 6, 256, 256)
# GPU memory 2405 MB
# 16.6 task/s (single 2080 ti)

# %% Mask (1, 6, 256, 256)
# GPU memory 2613 MB
# 46.0 task/s (single 2080 ti)

# %% RAFT (1, 6, 256, 256)
# GPU memory Failed
# 12.3 task/s (single 2080 ti)
