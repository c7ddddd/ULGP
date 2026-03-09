from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import torch

from .plugin import ULGPConfig, ULGPPlugin
from .vis import save_heatmap_png, save_npy


def _try_load_yaml(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        return json.loads(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ULGP Plugin CLI")
    p.add_argument("--a0", type=str, required=True, help="Path to anomaly map (.npy), shape (B,1,H,W) or (B,H,W)")
    p.add_argument("--feat", type=str, default="", help="Optional feature map (.npy), shape (B,C,h,w) or (B,N,C)")
    p.add_argument("--u", type=str, default="", help="Optional uncertainty map (.npy), shape (B,1,H,W) or (B,H,W)")
    p.add_argument("--img", type=str, default="", help="Optional image (.npy), shape (B,3,H,W)")
    p.add_argument("--out", type=str, required=True, help="Output refined .npy path")
    p.add_argument("--config", type=str, default="", help="YAML/JSON config path")
    p.add_argument("--mode", type=str, default="", choices=["", "auto", "full", "feature-lite", "map-only"])
    p.add_argument("--save_u", action="store_true", help="Save uncertainty map")
    p.add_argument("--save_candidate", action="store_true", help="Save candidate mask")
    p.add_argument("--save_vis", action="store_true", help="Save quick heatmap pngs")
    p.add_argument("--save_info", action="store_true", help="Save diagnostics json next to output")
    return p.parse_args()


def _load_optional(path: str) -> torch.Tensor | None:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.from_numpy(np.load(path))


def _build_config(args: argparse.Namespace) -> ULGPConfig:
    cfg_data = _try_load_yaml(args.config)
    if args.mode:
        cfg_data["mode"] = args.mode
    return ULGPConfig(**cfg_data)


def main() -> None:
    args = parse_args()
    cfg = _build_config(args)
    plugin = ULGPPlugin(config=cfg)

    a0 = torch.from_numpy(np.load(args.a0))
    feat = _load_optional(args.feat)
    u = _load_optional(args.u)
    img = _load_optional(args.img)

    refined, info = plugin.refine(a0, feature_map=feat, uncertainty_map=u, image=img, return_info=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_npy(args.out, refined)
    print(f"Saved refined map: {args.out}, shape={tuple(refined.shape)}")
    print(f"Mode={info['mode']}, graph_space={info['graph_space']}, mean_update={info['mean_update']:.6f}")

    base = os.path.splitext(args.out)[0]
    if args.save_info:
        with open(base + "_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
    if args.save_vis:
        save_heatmap_png(base + "_a0.png", a0)
        save_heatmap_png(base + "_ref.png", refined)
    if args.save_u or args.save_candidate:
        # Best-effort re-run internals for debug export through plugin methods.
        from .preprocess import prepare_plugin_inputs
        from .uncertainty import FeatureInstabilityUncertainty, PrecomputedUncertainty, ScoreVarianceUncertainty
        from .candidate import CandidateSelector

        bundle = prepare_plugin_inputs(a0, feature_map=feat, uncertainty_map=u, image=img)
        mode = info["mode"]
        cand = CandidateSelector(cfg.candidate_ratio, cfg.dilation_radius)(bundle.a0_graph).float()
        if mode == "full":
            u_map = PrecomputedUncertainty().compute(bundle.a0_graph, bundle.feat_graph, bundle.u_graph)
        elif mode == "feature-lite":
            u_map = FeatureInstabilityUncertainty(cfg.uncertainty_kernel_size).compute(bundle.a0_graph, bundle.feat_graph, bundle.u_graph)
        else:
            u_map = ScoreVarianceUncertainty(cfg.uncertainty_kernel_size).compute(bundle.a0_graph, bundle.feat_graph, bundle.u_graph)
        if args.save_u:
            save_npy(base + "_u.npy", u_map)
        if args.save_candidate:
            save_npy(base + "_candidate.npy", cand)


if __name__ == "__main__":
    main()

