import torch

from ulgp.uncertainty import FeatureInstabilityUncertainty, ScoreVarianceUncertainty


def test_score_uncertainty_shape():
    a0 = torch.rand(2, 1, 16, 16)
    u = ScoreVarianceUncertainty().compute(a0)
    assert u.shape == a0.shape


def test_feature_uncertainty_shape():
    a0 = torch.rand(2, 1, 16, 16)
    feat = torch.rand(2, 8, 8, 8)
    u = FeatureInstabilityUncertainty().compute(a0, feat=feat)
    assert u.shape == a0.shape
