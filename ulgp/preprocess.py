from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .utils import ensure_bchw, minmax_norm_per_image, to_feature_bchw


@dataclass
class InputBundle:
    a0_graph: torch.Tensor
    a0_output: torch.Tensor
    feat_graph: Optional[torch.Tensor]
    u_graph: Optional[torch.Tensor]
    img_graph: Optional[torch.Tensor]
    graph_space: str
    output_hw: tuple[int, int]
    output_dtype: torch.dtype
    output_device: torch.device
    input_was_3d: bool


def _to_work_tensor(x: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()


def prepare_plugin_inputs(
    anomaly_map: torch.Tensor,
    feature_map: Optional[torch.Tensor] = None,
    uncertainty_map: Optional[torch.Tensor] = None,
    image: Optional[torch.Tensor] = None,
) -> InputBundle:
    if anomaly_map.ndim not in (3, 4):
        raise ValueError(f"anomaly_map must be 3D/4D, got {tuple(anomaly_map.shape)}")
    orig_dtype = anomaly_map.dtype
    orig_device = anomaly_map.device
    was_3d = anomaly_map.ndim == 3

    a0_out = ensure_bchw(_to_work_tensor(anomaly_map, device=orig_device))
    b, _, h_out, w_out = a0_out.shape

    feat = _to_work_tensor(feature_map, device=orig_device)
    feat_bchw = None if feat is None else to_feature_bchw(feat)
    u = None if uncertainty_map is None else ensure_bchw(_to_work_tensor(uncertainty_map, device=orig_device))
    img = None if image is None else ensure_bchw(_to_work_tensor(image, device=orig_device))

    if feat_bchw is not None:
        if feat_bchw.shape[0] != b:
            raise ValueError("Batch size mismatch between anomaly_map and feature_map.")
        graph_space = "feature"
        gh, gw = feat_bchw.shape[-2:]
        a0_graph = F.interpolate(a0_out, size=(gh, gw), mode="bilinear", align_corners=False)
        u_graph = None if u is None else F.interpolate(u, size=(gh, gw), mode="bilinear", align_corners=False)
        img_graph = None if img is None else F.interpolate(img, size=(gh, gw), mode="bilinear", align_corners=False)
        feat_graph = feat_bchw
    else:
        graph_space = "map"
        a0_graph = a0_out
        feat_graph = None
        u_graph = u
        img_graph = img

    if u_graph is not None:
        u_graph = minmax_norm_per_image(u_graph)

    return InputBundle(
        a0_graph=a0_graph,
        a0_output=a0_out,
        feat_graph=feat_graph,
        u_graph=u_graph,
        img_graph=img_graph,
        graph_space=graph_space,
        output_hw=(h_out, w_out),
        output_dtype=orig_dtype,
        output_device=orig_device,
        input_was_3d=was_3d,
    )


def restore_output_format(refined_graph: torch.Tensor, bundle: InputBundle) -> torch.Tensor:
    x = refined_graph
    if x.shape[-2:] != bundle.output_hw:
        x = F.interpolate(x, size=bundle.output_hw, mode="bilinear", align_corners=False)
    x = x.to(device=bundle.output_device, dtype=bundle.output_dtype)
    return x.squeeze(1) if bundle.input_was_3d else x

