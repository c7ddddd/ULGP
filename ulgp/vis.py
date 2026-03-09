from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch


def save_npy(path: str, x: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, x.detach().cpu().numpy())


def save_heatmap_png(path: str, x: torch.Tensor, index: int = 0) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    arr = x.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[index, 0]
    elif arr.ndim == 3:
        arr = arr[index]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, cmap="jet")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

