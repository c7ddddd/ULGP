import torch

from ulgp import ULGPPlugin

a0 = torch.rand(1, 1, 64, 64)
f = torch.rand(1, 32, 16, 16)
u = torch.rand(1, 1, 64, 64)
plugin = ULGPPlugin(mode="full")
ref, info = plugin.refine(a0, feature_map=f, uncertainty_map=u, return_info=True)
print("full:", ref.shape, info["mode"], info["uncertainty_stats"])
