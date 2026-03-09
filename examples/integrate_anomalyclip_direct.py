from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ulgp import ULGPConfig, ULGPPlugin


def _load_npy(path: str | None) -> torch.Tensor | None:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.from_numpy(np.load(path))


def _maybe_fix_tokens(feat: torch.Tensor, drop_cls_if_needed: bool) -> torch.Tensor:
    # Common AnomalyCLIP case: (B, 1+L, C) with CLS token at index 0.
    if feat.ndim == 3 and drop_cls_if_needed:
        b, n, c = feat.shape
        h = int(round((n - 1) ** 0.5))
        if h * h == (n - 1):
            feat = feat[:, 1:, :]
    return feat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Direct ULGP integration for AnomalyCLIP outputs")
    p.add_argument("--a0", type=str, required=True, help="Anomaly map .npy from AnomalyCLIP, shape (B,1,H,W)/(B,H,W)")
    p.add_argument("--feat", type=str, default="", help="Optional feature .npy, shape (B,C,h,w) or (B,N,C)")
    p.add_argument("--u", type=str, default="", help="Optional uncertainty map .npy")
    p.add_argument("--img", type=str, default="", help="Optional input image .npy")
    p.add_argument("--out", type=str, required=True, help="Output refined anomaly map .npy")
    p.add_argument("--info_out", type=str, default="", help="Optional diagnostics json output path")
    p.add_argument("--mode", type=str, default="feature-lite", choices=["auto", "full", "feature-lite", "map-only"])
    p.add_argument("--drop_cls_if_needed", action="store_true", help="If feat is (B,1+L,C) and L is square, drop CLS token")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    a0 = _load_npy(args.a0)
    feat = _load_npy(args.feat)
    u = _load_npy(args.u)
    img = _load_npy(args.img)
    if a0 is None:
        raise ValueError("a0 is required")
    if feat is not None:
        feat = _maybe_fix_tokens(feat, drop_cls_if_needed=args.drop_cls_if_needed)

    cfg = ULGPConfig(mode=args.mode)
    plugin = ULGPPlugin(config=cfg)
    refined, info = plugin.refine(
        anomaly_map=a0,
        feature_map=feat,
        uncertainty_map=u,
        image=img,
        return_info=True,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, refined.detach().cpu().numpy())
    print(f"[ULGP] saved refined map: {args.out}, shape={tuple(refined.shape)}")
    print(
        "[ULGP] mode={mode} graph_space={space} mean_update={mu:.6f} gate_mean={gm:.6f} clamped_fraction={cf:.6f}".format(
            mode=info["mode"],
            space=info["graph_space"],
            mu=info["mean_update"],
            gm=info["gate_mean"],
            cf=info["clamped_fraction"],
        )
    )

    if args.info_out:
        os.makedirs(os.path.dirname(args.info_out) or ".", exist_ok=True)
        with open(args.info_out, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"[ULGP] saved info: {args.info_out}")


if __name__ == "__main__":
    main()

