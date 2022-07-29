from mmflow.apis import init_model, inference_model
from mmflow.datasets import visualize_flow, write_flow
from mmcv import flowwrite
import numpy as np

# Specify the path to model config and checkpoint file
config_file = 'configs/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.py'
checkpoint_file = 'checkpoints/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:1')

# test image pair, and save the results
img1 = 'my_data/thumos14/rawframes/train/v_BaseballPitch_g01_c01/img_00080.jpg'
img2 = 'my_data/thumos14/rawframes/train/v_BaseballPitch_g01_c01/img_00081.jpg'
result = inference_model(model, img1, img2)
np.save('raw_array.npy', result)
# save the optical flow file
write_flow(result, flow_file='mmflow_write_flow.flo')
flowwrite(result, filename='mmcv_flowwrite.jpg', quantize=True)
# save the visualized flow map
flow_map = visualize_flow(result, save_file='mmcv_flow2rgb.png')

# FlowNet2 failed with input resolution.
