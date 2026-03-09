import torch

from ulgp import ULGPPlugin


def test_map_only_shape():
    a0 = torch.rand(2, 1, 32, 32)
    plugin = ULGPPlugin(mode="map-only")
    out = plugin.refine(a0)
    assert out.shape == a0.shape


def test_feature_lite_shape():
    a0 = torch.rand(2, 1, 32, 32)
    feat = torch.rand(2, 16, 16, 16)
    plugin = ULGPPlugin(mode="feature-lite")
    out = plugin.refine(a0, feature_map=feat)
    assert out.shape == a0.shape


def test_full_info():
    a0 = torch.rand(1, 1, 16, 16)
    feat = torch.rand(1, 8, 8, 8)
    u = torch.rand(1, 1, 16, 16)
    plugin = ULGPPlugin(mode="full")
    out, info = plugin.refine(a0, feature_map=feat, uncertainty_map=u, return_info=True)
    assert out.shape == a0.shape
    assert "mean_update" in info
