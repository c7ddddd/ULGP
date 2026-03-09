import torch

from ulgp.graph import FeatureKNNGraphBuilder, GridGraphBuilder


def test_grid_graph_build():
    m = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    m[:, :, 2:6, 2:6] = True
    builder = GridGraphBuilder()
    graphs = builder.build(m, feat=None, image=None)
    assert len(graphs) == 1
    assert graphs[0].P.shape[0] == int(m.sum())


def test_feature_graph_build():
    m = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    m[:, :, 1:7, 1:7] = True
    feat = torch.rand(1, 4, 8, 8)
    builder = FeatureKNNGraphBuilder(k=4)
    graphs = builder.build(m, feat=feat, image=None)
    assert graphs[0].P.shape[0] == int(m.sum())
