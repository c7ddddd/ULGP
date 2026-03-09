from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class FusionStrategy(ABC):
    @abstractmethod
    def fuse(self, a0: torch.Tensor, a_tmp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearFusion(FusionStrategy):
    def __init__(self, fusion_lambda: float = 0.7):
        self.fusion_lambda = float(fusion_lambda)

    def fuse(self, a0: torch.Tensor, a_tmp: torch.Tensor) -> torch.Tensor:
        lam = self.fusion_lambda
        return lam * a_tmp + (1.0 - lam) * a0


class IdentityFusion(FusionStrategy):
    def fuse(self, a0: torch.Tensor, a_tmp: torch.Tensor) -> torch.Tensor:
        return a_tmp

