import torch

from ulgp import ULGPPlugin

a0 = torch.rand(1, 1, 64, 64)
f = torch.rand(1, 32, 16, 16)
plugin = ULGPPlugin(mode="feature-lite")
ref, info = plugin.refine(a0, feature_map=f, return_info=True)
print("feature-lite:", ref.shape, info["mode"], info["num_nodes"])
