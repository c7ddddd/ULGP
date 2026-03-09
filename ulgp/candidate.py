from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import ensure_bchw


def topk_candidate_mask(a0: torch.Tensor, ratio: float = 0.05) -> torch.Tensor:
    a0 = ensure_bchw(a0)
    b, _, h, w = a0.shape
    flat = a0.view(b, -1)
    k = max(1, int(flat.shape[1] * ratio))
    vals, _ = torch.topk(flat, k=k, dim=1, largest=True, sorted=False)
    th = vals.amin(dim=1, keepdim=True)
    mask = flat >= th
    return mask.view(b, 1, h, w)


def dilate_mask(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    mask = ensure_bchw(mask.float())
    if radius <= 0:
        return mask > 0.5
    k = 2 * radius + 1
    out = F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)
    return out > 0.5


class CandidateSelector:
    def __init__(self, ratio: float = 0.05, dilation_radius: int = 1):
        self.ratio = ratio
        self.dilation_radius = dilation_radius

    def __call__(self, a0: torch.Tensor) -> torch.Tensor:
        cand = topk_candidate_mask(a0, ratio=self.ratio)
        return dilate_mask(cand, radius=self.dilation_radius)
