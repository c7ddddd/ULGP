from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x
    if x.ndim == 3:
        return x.unsqueeze(1)
    raise ValueError(f"Expected tensor with 3 or 4 dims, got shape={tuple(x.shape)}")


def minmax_norm_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = ensure_bchw(x)
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def to_feature_bchw(feat: torch.Tensor, out_hw: Tuple[int, int] | None = None) -> torch.Tensor:
    if feat.ndim == 4:
        f = feat
    elif feat.ndim == 3:
        b, n, c = feat.shape
        h = int(round(n ** 0.5))
        if h * h != n:
            raise ValueError(f"Cannot reshape (B,N,C) with N={n} to square map.")
        f = feat.permute(0, 2, 1).contiguous().view(b, c, h, h)
    else:
        raise ValueError(f"Expected feature map with 3 or 4 dims, got shape={tuple(feat.shape)}")
    if out_hw is not None and (f.shape[-2], f.shape[-1]) != out_hw:
        f = F.interpolate(f, size=out_hw, mode="bilinear", align_corners=False)
    return f

