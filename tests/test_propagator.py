import torch

from ulgp.graph import GraphBatch
from ulgp.propagator import propagate_scores


def test_propagator_clamp():
    a0 = torch.rand(1, 1, 8, 8)
    u = torch.zeros_like(a0)
    idx = torch.arange(8 * 8)
    p = torch.eye(8 * 8)
    graphs = [GraphBatch(node_idx=idx, P=p, edge_count=0)]
    out, info = propagate_scores(a0, u, graphs, clamp_delta=0.01, steps=2)
    d = (out - a0).abs().max().item()
    assert d <= 0.011
    assert info["propagation_steps"] == 2
