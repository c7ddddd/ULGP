"""
Pseudo integration snippet for MuSc-like pipelines.
Replace `A0/F/U/I` with actual tensors in your inference loop.
"""
from ulgp import ULGPPlugin

plugin = ULGPPlugin(mode="feature-lite")

# A0: anomaly map from baseline model
# F: aligned feature map from baseline model
# U: optional uncertainty map
# I: optional image
# A_ref = plugin.refine(A0, feature_map=F, uncertainty_map=U, image=I)
