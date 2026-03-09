import torch

from ulgp import ULGPPlugin

a0 = torch.rand(1, 1, 64, 64)
plugin = ULGPPlugin(mode="map-only")
ref = plugin.refine(a0)
print("map-only:", ref.shape)
