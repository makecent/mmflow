import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from mmflow.apis import init_model

# Specify the path to model config and checkpoint file
config_file = 'configs/raft_8x2_100k_mixed_368x768.py'
checkpoint_file = 'checkpoints/raft_8x2_100k_mixed_368x768.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# benchmark flops and params
# requires revising the test_mode to False, i.e., to training mode because testing output numpy array cannot be traced.
flops = FlopCountAnalysis(model, (torch.rand(1, 6, 256, 256).cuda(), torch.rand(1, 2, 256, 256).cuda(), None))
# print(flop_count_table(flops, max_depth=10))
params = parameter_count(model)

print(f"GFLOPS:\t{flops.total() / 1e9:.2f} G")
print(f"Params:\t{params[''] / 1e6:.2f} M")

# GMA 224x
# GFLOPS:	45.62 G
# Params:	5.80 M

# PWC 256x
# GFLOPS:	12.02 G
# Params:	9.37 M

# FlowNet2 256x
# GFLOPS:	18.61 G
# Params:   116.57 M

# LiteFlowNet2 256x
# GFLOPS:	9.70 G
# Params:   6.43 M

# IRR 256x
# GFLOPS:	7.59 G
# Params:   6.36 M

# Mask 256x
# GFLOPS:	24.42 G
# Params:   20.66 M

# RAFT 256x
# GFLOPS:	52.04 G
# Params:   5.26 M

