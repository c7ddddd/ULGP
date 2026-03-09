from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from .utils import ensure_bchw, minmax_norm_per_image, to_feature_bchw


def _local_var_map(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    x = ensure_bchw(x)
    pad = kernel_size // 2
    mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    mean2 = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=pad)
    return torch.relu(mean2 - mean * mean)


class UncertaintyStrategy(ABC):
    @abstractmethod
    def compute(self, a0: torch.Tensor, feat: torch.Tensor | None = None, u: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class PrecomputedUncertainty(UncertaintyStrategy):
    def compute(self, a0: torch.Tensor, feat: torch.Tensor | None = None, u: torch.Tensor | None = None) -> torch.Tensor:
        if u is None:
            raise ValueError("PrecomputedUncertainty requires uncertainty map `u`.")
        return minmax_norm_per_image(ensure_bchw(u))


class ScoreVarianceUncertainty(UncertaintyStrategy):
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size

    def compute(self, a0: torch.Tensor, feat: torch.Tensor | None = None, u: torch.Tensor | None = None) -> torch.Tensor:
        out = _local_var_map(a0, kernel_size=self.kernel_size)
        return minmax_norm_per_image(out)


class FeatureInstabilityUncertainty(UncertaintyStrategy):
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size

    def compute(self, a0: torch.Tensor, feat: torch.Tensor | None = None, u: torch.Tensor | None = None) -> torch.Tensor:
        if feat is None:
            raise ValueError("FeatureInstabilityUncertainty requires `feat`.")
        f = to_feature_bchw(feat, out_hw=a0.shape[-2:])
        f = F.normalize(f, p=2, dim=1)
        out = _local_var_map(f, kernel_size=self.kernel_size).mean(dim=1, keepdim=True)
        return minmax_norm_per_image(out)

