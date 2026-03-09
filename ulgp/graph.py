from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from .utils import to_feature_bchw


@dataclass
class GraphBatch:
    node_idx: torch.Tensor
    P: torch.Tensor
    edge_count: int


def _boundary_strength(image_bchw: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if image_bchw is None:
        return None
    gray = image_bchw.mean(dim=1, keepdim=True)
    gx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
    gy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
    g = torch.sqrt(gx * gx + gy * gy + 1e-12)
    g = g / (g.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return g


class GraphBuilder(ABC):
    @abstractmethod
    def build(self, candidate_mask: torch.Tensor, feat: Optional[torch.Tensor], image: Optional[torch.Tensor]) -> List[GraphBatch]:
        raise NotImplementedError


class FeatureKNNGraphBuilder(GraphBuilder):
    def __init__(self, k: int = 8, tau_b: float = 0.25):
        self.k = k
        self.tau_b = tau_b

    def build(self, candidate_mask: torch.Tensor, feat: Optional[torch.Tensor], image: Optional[torch.Tensor]) -> List[GraphBatch]:
        if feat is None:
            raise ValueError("FeatureKNNGraphBuilder requires feature_map.")
        b, _, h, w = candidate_mask.shape
        feat_bchw = to_feature_bchw(feat, out_hw=(h, w))
        feat_flat = F.normalize(feat_bchw.permute(0, 2, 3, 1).reshape(b, h * w, -1), p=2, dim=-1)
        boundary = _boundary_strength(image)
        boundary_flat = None if boundary is None else boundary.reshape(b, -1)

        graphs: List[GraphBatch] = []
        for bi in range(b):
            node_idx = torch.where(candidate_mask[bi, 0].reshape(-1))[0]
            n = int(node_idx.numel())
            if n == 0:
                graphs.append(GraphBatch(node_idx=node_idx, P=torch.zeros((0, 0), device=candidate_mask.device), edge_count=0))
                continue

            f = feat_flat[bi][node_idx]
            sim = torch.matmul(f, f.t()).clamp_min(0.0)
            if boundary_flat is not None:
                bval = boundary_flat[bi][node_idx]
                bdiff = (bval[:, None] - bval[None, :]).abs()
                sim = sim * torch.exp(-bdiff / max(self.tau_b, 1e-6))
            sim.fill_diagonal_(0.0)
            kk = min(self.k, max(1, n - 1))
            if n > 1:
                vals, inds = torch.topk(sim, k=kk, dim=1, largest=True, sorted=False)
                p = torch.zeros_like(sim)
                p.scatter_(1, inds, vals)
                edges = int((vals > 0).sum().item())
            else:
                p = torch.zeros_like(sim)
                edges = 0
            p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-6)
            graphs.append(GraphBatch(node_idx=node_idx, P=p, edge_count=edges))
        return graphs


class GridGraphBuilder(GraphBuilder):
    def __init__(self, tau_b: float = 0.25):
        self.tau_b = tau_b

    def build(self, candidate_mask: torch.Tensor, feat: Optional[torch.Tensor], image: Optional[torch.Tensor]) -> List[GraphBatch]:
        b, _, h, w = candidate_mask.shape
        boundary = _boundary_strength(image)
        boundary_flat = None if boundary is None else boundary.reshape(b, -1)
        graphs: List[GraphBatch] = []
        for bi in range(b):
            node_idx = torch.where(candidate_mask[bi, 0].reshape(-1))[0]
            n = int(node_idx.numel())
            if n == 0:
                graphs.append(GraphBatch(node_idx=node_idx, P=torch.zeros((0, 0), device=candidate_mask.device), edge_count=0))
                continue
            yy = (node_idx // w).float()
            xx = (node_idx % w).float()
            dist = (yy[:, None] - yy[None, :]).abs() + (xx[:, None] - xx[None, :]).abs()
            p = torch.where(dist <= 1.5, torch.ones_like(dist), torch.zeros_like(dist))
            p.fill_diagonal_(0.0)
            if boundary_flat is not None:
                bval = boundary_flat[bi][node_idx]
                bdiff = (bval[:, None] - bval[None, :]).abs()
                p = p * torch.exp(-bdiff / max(self.tau_b, 1e-6))
            edges = int((p > 0).sum().item())
            p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-6)
            graphs.append(GraphBatch(node_idx=node_idx, P=p, edge_count=edges))
        return graphs

