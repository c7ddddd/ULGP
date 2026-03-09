import torch

from ulgp.candidate import CandidateSelector


def test_candidate_non_empty():
    a0 = torch.rand(1, 1, 16, 16)
    sel = CandidateSelector(ratio=0.05, dilation_radius=1)
    mask = sel(a0)
    assert mask.shape == a0.shape
    assert mask.any()
