"""
Pseudo integration snippet for AnomalyCLIP-like pipelines.
Replace `A0/F/U/I` with actual tensors in your inference loop.
"""
from ulgp import ULGPPlugin

plugin = ULGPPlugin(mode="full")

# A0: anomaly map
# F: CLIP patch/token feature map
# U: optional uncertainty map from branch disagreement
# I: optional image
# A_ref = plugin.refine(A0, feature_map=F, uncertainty_map=U, image=I)
