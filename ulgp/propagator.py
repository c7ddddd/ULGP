from __future__ import annotations

import torch

from .graph import GraphBatch


def propagate_scores(
    a0: torch.Tensor,
    u: torch.Tensor,
    graphs: list[GraphBatch],
    alpha: float = 0.5,
    beta: float = 0.1,
    tau_u: float = 0.5,
    steps: int = 3,
    clamp_delta: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    b, _, h, w = a0.shape
    out = a0.clone()
    a0_flat = a0.reshape(b, -1)
    u_flat = u.reshape(b, -1)

    total_nodes = 0
    total_edges = 0
    max_update = 0.0
    gate_sum = 0.0
    gate_count = 0
    clamped_count = 0
    updated_count = 0
    for bi, graph in enumerate(graphs):
        idx = graph.node_idx
        if idx.numel() == 0:
            continue
        total_nodes += int(idx.numel())
        total_edges += int(graph.edge_count)
        p = graph.P
        s0 = a0_flat[bi][idx]
        s = s0.clone()
        ui = u_flat[bi][idx]
        g = torch.exp(-ui / max(tau_u, 1e-6))
        gate_sum += float(g.sum().item())
        gate_count += int(g.numel())

        for _ in range(steps):
            s_prop = p @ s
            s_new = (1.0 - alpha * g - beta) * s + (alpha * g) * s_prop + beta * s0
            lo = s0 - clamp_delta
            hi = s0 + clamp_delta
            s_clip = torch.clamp(s_new, min=lo, max=hi)
            clamped_count += int(((s_new < lo) | (s_new > hi)).sum().item())
            updated_count += int(s_new.numel())
            s = s_clip

        out.view(b, -1)[bi, idx] = s
        upd = (s - s0).abs()
        max_update = max(max_update, float(upd.max().item()))

    out = out.view(b, 1, h, w)
    mean_update = float((out - a0).abs().mean().item())
    info = {
        "num_nodes": total_nodes,
        "num_edges": total_edges,
        "mean_update": mean_update,
        "max_update": max_update,
        "propagation_steps": int(steps),
        "gate_mean": (gate_sum / gate_count) if gate_count > 0 else 0.0,
        "clamped_fraction": (clamped_count / updated_count) if updated_count > 0 else 0.0,
    }
    return out, info
