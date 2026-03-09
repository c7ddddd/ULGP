from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import torch

from .candidate import CandidateSelector
from .fusion import FusionStrategy, IdentityFusion, LinearFusion
from .graph import FeatureKNNGraphBuilder, GraphBuilder, GridGraphBuilder
from .preprocess import prepare_plugin_inputs, restore_output_format
from .propagator import propagate_scores
from .uncertainty import (
    FeatureInstabilityUncertainty,
    PrecomputedUncertainty,
    ScoreVarianceUncertainty,
    UncertaintyStrategy,
)


@dataclass
class ULGPConfig:
    candidate_ratio: float = 0.05
    dilation_radius: int = 1
    k: int = 8
    num_steps: int = 3
    alpha: float = 0.5
    beta: float = 0.1
    tau_u: float = 0.5
    clamp_delta: float = 0.05
    fusion_lambda: float = 0.7
    tau_b: float = 0.25
    uncertainty_kernel_size: int = 3
    mode: str = "auto"  # auto|full|feature-lite|map-only
    fusion: str = "linear"  # linear|identity


class ULGPPlugin:
    def __init__(self, config: Optional[ULGPConfig] = None, **kwargs):
        self.cfg = config or ULGPConfig(**kwargs)
        self.selector = CandidateSelector(self.cfg.candidate_ratio, self.cfg.dilation_radius)

    def _resolve_mode(self, feat: Optional[torch.Tensor], u: Optional[torch.Tensor]) -> str:
        mode = self.cfg.mode
        if mode != "auto":
            return mode
        if feat is not None and u is not None:
            return "full"
        if feat is not None:
            return "feature-lite"
        return "map-only"

    def _build_graph_policy(self, mode: str) -> GraphBuilder:
        if mode in ("full", "feature-lite"):
            return FeatureKNNGraphBuilder(k=self.cfg.k, tau_b=self.cfg.tau_b)
        return GridGraphBuilder(tau_b=self.cfg.tau_b)

    def _build_uncertainty_policy(self, mode: str) -> UncertaintyStrategy:
        if mode == "full":
            return PrecomputedUncertainty()
        if mode == "feature-lite":
            return FeatureInstabilityUncertainty(kernel_size=self.cfg.uncertainty_kernel_size)
        return ScoreVarianceUncertainty(kernel_size=self.cfg.uncertainty_kernel_size)

    def _build_fusion_policy(self) -> FusionStrategy:
        if self.cfg.fusion == "identity":
            return IdentityFusion()
        return LinearFusion(fusion_lambda=self.cfg.fusion_lambda)

    @torch.no_grad()
    def refine(
        self,
        anomaly_map: torch.Tensor,
        feature_map: Optional[torch.Tensor] = None,
        uncertainty_map: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ):
        t0 = perf_counter()
        bundle = prepare_plugin_inputs(
            anomaly_map=anomaly_map,
            feature_map=feature_map,
            uncertainty_map=uncertainty_map,
            image=image,
        )
        mode = self._resolve_mode(bundle.feat_graph, bundle.u_graph)
        graph_builder = self._build_graph_policy(mode)
        u_policy = self._build_uncertainty_policy(mode)
        fusion_policy = self._build_fusion_policy()

        cand = self.selector(bundle.a0_graph)
        actual_ratio = float(cand.float().mean().item())
        u_graph = u_policy.compute(
            a0=bundle.a0_graph,
            feat=bundle.feat_graph,
            u=bundle.u_graph,
        )
        graphs = graph_builder.build(
            candidate_mask=cand,
            feat=bundle.feat_graph,
            image=bundle.img_graph,
        )
        a_tmp, pinfo = propagate_scores(
            a0=bundle.a0_graph,
            u=u_graph,
            graphs=graphs,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            tau_u=self.cfg.tau_u,
            steps=self.cfg.num_steps,
            clamp_delta=self.cfg.clamp_delta,
        )
        a_graph_ref = fusion_policy.fuse(bundle.a0_graph, a_tmp)
        a_ref = restore_output_format(a_graph_ref, bundle)
        if not return_info:
            return a_ref

        info = {
            "mode": mode,
            "graph_space": bundle.graph_space,
            "candidate_ratio_actual": actual_ratio,
            "uncertainty_stats": {
                "min": float(u_graph.min().item()),
                "mean": float(u_graph.mean().item()),
                "max": float(u_graph.max().item()),
            },
            "runtime_ms": float((perf_counter() - t0) * 1000.0),
        }
        info.update(pinfo)
        return a_ref, info

