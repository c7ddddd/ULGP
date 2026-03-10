"""
UAR-SRA (Uncertainty-Aware Region-anchored Self-Refinement by Anchoring) for AnomalyCLIP

Usage:
    cd /home/dct/unified_model/AnomalyCLIP-main
    python experiments/uar_sra.py --dataset mvtec --class_name ALL --device 0
"""

import os
import sys
import time
import sys

ULGP_ROOT = '/home/dct/unified_model/ULGP'
if ULGP_ROOT not in sys.path:
    sys.path.insert(0, ULGP_ROOT)
import random
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import AnomalyCLIP_lib
from AnomalyCLIP_lib.model_load import load as load_clip_model
from AnomalyCLIP_lib.AnomalyCLIP import AnomalyCLIP
from prompt_ensemble import AnomalyCLIP_PromptLearner
from metrics import image_level_metrics, pixel_level_metrics
from dataset import Dataset  # 官方 dataset
from utils import get_transform


_CLASSNAMES_mvtec = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]
_CLASSNAMES_visa = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4",
    "pipe_fryum",
]


# ================================================================
# General helpers
# ================================================================

def seed_everything(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================================================================
# Metrics (对齐官方: pixel_auroc, pixel_aupro, image_auroc, image_ap)
# ================================================================

def gaussian_smooth(pr_px, sigma=4):
    smoothed = np.zeros_like(pr_px)
    for i in range(pr_px.shape[0]):
        smoothed[i] = gaussian_filter(pr_px[i], sigma=sigma)
    return smoothed


def image_score_from_map(pr_px, mode="topk", topk_ratio=0.01, mix_alpha=0.5):
    n = pr_px.shape[0]
    flat = pr_px.reshape(n, -1)
    if mode == "max":
        return flat.max(axis=1)
    if mode == "topk":
        k = max(1, int(flat.shape[1] * topk_ratio))
        part = np.partition(flat, -k, axis=1)[:, -k:]
        return part.mean(axis=1)
    if mode == "mix":
        k = max(1, int(flat.shape[1] * topk_ratio))
        part = np.partition(flat, -k, axis=1)[:, -k:]
        s_topk = part.mean(axis=1)
        s_max = flat.max(axis=1)
        return mix_alpha * s_max + (1.0 - mix_alpha) * s_topk
    raise ValueError(f"Unknown image_score_mode: {mode}")


def image_score_topk_torch(a_map, topk_ratio=0.01):
    bsz = a_map.shape[0]
    flat = a_map.view(bsz, -1)
    k = max(1, int(flat.shape[1] * topk_ratio))
    return torch.topk(flat, k=k, dim=1).values.mean(dim=1)


def image_score_from_map_torch(a_map, mode="topk", topk_ratio=0.01, mix_alpha=0.5):
    bsz = a_map.shape[0]
    flat = a_map.view(bsz, -1)
    if mode == "max":
        return flat.max(dim=1).values
    if mode == "topk":
        k = max(1, int(flat.shape[1] * topk_ratio))
        return torch.topk(flat, k=k, dim=1).values.mean(dim=1)
    if mode == "mix":
        k = max(1, int(flat.shape[1] * topk_ratio))
        s_topk = torch.topk(flat, k=k, dim=1).values.mean(dim=1)
        s_max = flat.max(dim=1).values
        return mix_alpha * s_max + (1.0 - mix_alpha) * s_topk
    raise ValueError(f"Unknown image_score_mode: {mode}")


def sampled_global_quantiles(x, q_low=0.10, q_high=0.97, max_samples=500000, seed=111):
    flat = x.reshape(-1)
    if flat.numel() <= max_samples:
        samp = flat
    else:
        # Use CPU generator for broader PyTorch version compatibility.
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randint(0, flat.numel(), (max_samples,), device=flat.device, generator=g)
        samp = flat[idx]
    ql = torch.quantile(samp, q_low)
    qh = torch.quantile(samp, q_high)
    return ql, qh


def compute_metrics(
    gt_sp, gt_px, pr_px, pr_sp=None, sigma=4,
    image_score_mode="topk", image_score_topk_ratio=0.01, image_score_mix_alpha=0.5,
):
    """
    Evaluate with official metrics.py APIs:
      image_level_metrics / pixel_level_metrics
    """
    pr_px_smooth = gaussian_smooth(pr_px, sigma=sigma)
    if pr_sp is None:
        pr_sp = image_score_from_map(
            pr_px_smooth,
            mode=image_score_mode,
            topk_ratio=image_score_topk_ratio,
            mix_alpha=image_score_mix_alpha,
        )

    results = {
        "_tmp": {
            "gt_sp": np.array(gt_sp),
            "pr_sp": np.array(pr_sp),
            "imgs_masks": np.array(gt_px),
            "anomaly_maps": np.array(pr_px_smooth),
        }
    }

    m = {"image_auroc": 0.0, "image_ap": 0.0, "pixel_auroc": 0.0, "pixel_aupro": 0.0}

    # Keep behavior robust when labels are degenerate.
    if len(np.unique(results["_tmp"]["gt_sp"])) >= 2:
        m["image_auroc"] = image_level_metrics(results, "_tmp", "image-auroc")
        m["image_ap"] = image_level_metrics(results, "_tmp", "image-ap")

    gt_arr = results["_tmp"]["imgs_masks"]
    if np.asarray(gt_arr).sum() > 0:
        m["pixel_auroc"] = pixel_level_metrics(results, "_tmp", "pixel-auroc")
        m["pixel_aupro"] = pixel_level_metrics(results, "_tmp", "pixel-aupro")

    return m

# ================================================================
# Uncertainty helpers
# ================================================================

def normalize_01_per_image(x, q_low=0.05, q_high=0.95, eps=1e-8):
    B = x.shape[0]
    flat = x.view(B, -1)
    ql = torch.quantile(flat, q_low, dim=1, keepdim=True).view(B, 1, 1, 1)
    qh = torch.quantile(flat, q_high, dim=1, keepdim=True).view(B, 1, 1, 1)
    y = (x - ql) / (qh - ql + eps)
    return y.clamp(0.0, 1.0)


def compute_uncertainty_from_layers(amaps):
    st = torch.stack(amaps, dim=0)
    return torch.var(st, dim=0, unbiased=False)


def tta_flips(images):
    return [
        ("id", lambda x: x, lambda y: y),
        ("h",  lambda x: x.flip(-1), lambda y: y.flip(-1)),
        ("v",  lambda x: x.flip(-2), lambda y: y.flip(-2)),
        ("hv", lambda x: x.flip(-1).flip(-2), lambda y: y.flip(-1).flip(-2)),
    ]


# ================================================================
# Region helpers
# ================================================================

def kmeans_torch(x, k, n_iter=10):
    L, C = x.shape
    idx = torch.randperm(L, device=x.device)[:k]
    centers = x[idx].clone()
    for _ in range(n_iter):
        dist = 2.0 - 2.0 * (x @ centers.t())
        labels = dist.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k).float().to(x.device)
        empty = (counts < 1)
        counts = counts.clamp(min=1.0).unsqueeze(1)
        new_centers = new_centers / counts
        new_centers = F.normalize(new_centers, dim=1)
        if empty.any():
            ridx = torch.randperm(L, device=x.device)[:empty.sum().item()]
            new_centers[empty] = x[ridx]
        centers = new_centers
    return labels


@torch.no_grad()
def build_regions_from_clip_feat(feats, n_regions=64, n_iter=10, device="cuda"):
    N, C, h, w = feats.shape
    labels_all = torch.empty(N, h, w, dtype=torch.int64)
    for i in tqdm(range(N), desc="build_regions"):
        f = feats[i].to(device)
        x = f.view(C, -1).t().contiguous()
        x = F.normalize(x, dim=1)
        lab = kmeans_torch(x, n_regions, n_iter=n_iter)
        labels_all[i] = lab.view(h, w).cpu()
    return labels_all


def build_grid_regions(h, w, grid=8):
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ry = (yy * grid // h)
    rx = (xx * grid // w)
    return (ry * grid + rx).long()


def region_var_loss(a_pred, labels_patch, u_map=None, tau_r=0.5, eps=1e-8):
    B = a_pred.shape[0]
    h, w = labels_patch.shape[-2], labels_patch.shape[-1]
    a_patch = F.interpolate(a_pred, size=(h, w), mode="bilinear", align_corners=True)
    a_flat = a_patch.squeeze(1).reshape(B, -1)
    lab_flat = labels_patch.reshape(B, -1)

    if u_map is not None:
        u_patch = F.interpolate(u_map, size=(h, w), mode="bilinear", align_corners=True)
        u_flat = u_patch.squeeze(1).reshape(B, -1).clamp(0.0, 1.0)
    else:
        u_flat = None

    K = int(lab_flat.max().item()) + 1
    ones = torch.ones_like(a_flat)
    sum_a = torch.zeros(B, K, device=a_pred.device).scatter_add_(1, lab_flat, a_flat)
    sum_a2 = torch.zeros(B, K, device=a_pred.device).scatter_add_(1, lab_flat, a_flat * a_flat)
    cnt = torch.zeros(B, K, device=a_pred.device).scatter_add_(1, lab_flat, ones).clamp(min=1.0)
    mu = sum_a / cnt
    var = (sum_a2 / cnt - mu * mu).clamp(min=0.0)

    if u_flat is not None:
        sum_u = torch.zeros(B, K, device=a_pred.device).scatter_add_(1, lab_flat, u_flat)
        u_mu = (sum_u / cnt).clamp(0.0, 1.0)
        wt = torch.exp(-u_mu / max(tau_r, 1e-6))
        loss = (wt * var).sum() / (wt.sum() + eps)
    else:
        loss = var.mean()
    return loss


# ================================================================
# AnomalyCLIP Scorer — 完全对齐官方 test.py
# ================================================================

class AnomalyCLIPScorer:
    """
    严格复用官方 test.py 的推理逻辑，
    包括 features_list、feature_map_layer、anomaly map 融合方式。
    """
    def __init__(
        self,
        device,
        image_size=518,
        depth=9,
        n_ctx=12,
        t_n_ctx=4,
        checkpoint=None,
        features_list=(24,),  # align with your test.sh default
        feature_map_layer=(0, 1, 2, 3),  # 官方默认 [0,1,2,3]
        sigma=4,
    ):
        self.device = device
        self.image_size = image_size
        self.features_list = list(features_list)
        self.feature_map_layer = list(feature_map_layer)
        self.sigma = sigma

        design_details = {
            "trainer": "MaPLe",
            "vision_depth": 0,
            "language_depth": depth,
            "vision_ctx": 0,
            "language_ctx": n_ctx,
            "maple_length": n_ctx,
            "learnabel_text_embedding_depth": depth,
            "learnabel_text_embedding_length": t_n_ctx,
            "Prompt_length": n_ctx,
        }

        self.model, _ = load_clip_model(
            "ViT-L/14@336px", device=device, design_details=design_details
        )
        self.model.eval()
        self.model.visual.DAPM_replace(DPAM_layer=20)

        # 需要在 cpu 上创建 prompt learner，然后一起搬到 device
        self.model.to("cpu")
        self.prompt_learner = AnomalyCLIP_PromptLearner(
            self.model, design_details=design_details
        )
        self.model.to(device)

        if checkpoint and os.path.exists(checkpoint):
            ckpt = torch.load(checkpoint, map_location="cpu")
            if "prompt_learner" in ckpt:
                self.prompt_learner.load_state_dict(ckpt["prompt_learner"])
            print(f"  Loaded: {checkpoint}")

        self.prompt_learner.to(device)
        self._precompute_text()

    def _precompute_text(self):
        with torch.no_grad():
            prompts, tok, compound = self.prompt_learner(cls_id=None)
            prompts = prompts.to(self.device)
            tok = tok.to(self.device)
            compound = [t.to(self.device) for t in compound]
            tf = self.model.encode_text_learn(prompts, tok, compound).float()
            # Match official test.py: split into normal/abnormal groups.
            self.text_features = torch.stack(torch.chunk(tf, dim=0, chunks=2), dim=1)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        print(f"  text_features shape: {self.text_features.shape}")

    @torch.no_grad()
    def score_batch(self, images, cls_id=None, return_amaps=False):
        """
        对齐官方 test.py 的推理流程。
        images: (B,3,H,W) on device
        返回: a0 (B,1,H,W), clip_feat (B,C,h,h)
        如果 return_amaps=True，额外返回每个 feature_map_layer 的 amap list
        """
        bsz = images.shape[0]

        with torch.amp.autocast("cuda"):
            image_features, patch_tokens = self.model.encode_image(
                images, self.features_list, DPAM_layer=20
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            if cls_id is None:
                cls_idx = 0
            else:
                cls_idx = int(cls_id)
                cls_idx = max(0, min(cls_idx, self.text_features.shape[0] - 1))
            class_text = self.text_features[cls_idx]  # (2, C)

            # Per-image abnormal probability for selected class.
            text_logits = image_features @ class_text.t()  # (B, 2)
            text_probs = (text_logits / 0.07).softmax(dim=-1)[:, 1]

            anomaly_maps = []
            if len(self.feature_map_layer) > 0:
                selected_layers = []
                for li in self.feature_map_layer:
                    li = int(li)
                    if 0 <= li < len(patch_tokens):
                        selected_layers.append(li)
                if len(selected_layers) == 0:
                    selected_layers = list(range(len(patch_tokens)))
            else:
                selected_layers = list(range(len(patch_tokens)))

            for idx in selected_layers:
                patch_feature = patch_tokens[idx]
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, class_text)
                similarity_map = AnomalyCLIP_lib.get_similarity_map(
                    similarity[:, 1:, :], self.image_size
                )
                anomaly_map = (similarity_map[..., 1] + 1.0 - similarity_map[..., 0]) / 2.0
                anomaly_maps.append(anomaly_map.unsqueeze(1).float())

            a0 = torch.stack(anomaly_maps).sum(dim=0)

            last_layer_idx = selected_layers[-1]
            lf = patch_tokens[last_layer_idx][:, 1:, :]

            L = lf.shape[1]
            hh = int(np.sqrt(L))
            clip_feat = lf.permute(0, 2, 1).view(bsz, -1, hh, hh).float()

        if return_amaps:
            return a0, clip_feat, anomaly_maps, text_probs
        return a0, clip_feat, text_probs


# ================================================================
# Uncertainty-aware Anchor Miner
# ================================================================

class SoftAnchorMiner:
    def __init__(self, confidence_floor=0.55, u0=0.7, tau_u=0.3, global_alpha=0.5):
        self.confidence_floor = confidence_floor
        self.u0 = u0
        self.tau_u = tau_u
        self.global_alpha = global_alpha

    def mine_confident(self, amap, progress, u_map=None, ql_global=None, qh_global=None):
        bsz, _, hh, ww = amap.shape
        flat = amap.view(bsz, -1)
        q_high = 99.7 - 2.0 * progress
        q_low = 5.0 + 15.0 * progress
        temp = 0.08 + 0.08 * progress
        ql_img = torch.quantile(flat, q_low / 100.0, dim=1, keepdim=True)
        qh_img = torch.quantile(flat, q_high / 100.0, dim=1, keepdim=True)
        if ql_global is not None and qh_global is not None:
            ql = (1.0 - self.global_alpha) * ql_img + self.global_alpha * ql_global
            qh = (1.0 - self.global_alpha) * qh_img + self.global_alpha * qh_global
        else:
            ql, qh = ql_img, qh_img
        scale = (qh - ql).clamp(min=1e-6)
        w_n = torch.sigmoid((ql - flat) / (temp * scale))
        w_a = torch.sigmoid((flat - qh) / (temp * scale))
        confidence = (w_n + w_a).clamp(0.0, 1.0)

        if u_map is not None:
            u_flat = u_map.view(bsz, -1).clamp(0.0, 1.0)
            g = torch.exp(-u_flat / max(self.tau_u, 1e-6))
            w_n = w_n * g
            w_a = w_a * g
            valid_u = (u_flat < self.u0).float()
        else:
            valid_u = 1.0

        valid = (confidence > self.confidence_floor).float() * valid_u
        w_n = (w_n * valid).view(bsz, 1, hh, ww)
        w_a = (w_a * valid).view(bsz, 1, hh, ww)
        valid_ratio = valid.mean().item()
        return w_n, w_a, valid_ratio


# ================================================================
# RefineNet
# ================================================================

class RefineNet(nn.Module):
    def __init__(self, feat_dim=64, clip_dim=1024, use_u=True, delta_clamp=0.1):
        super().__init__()
        self.use_u = use_u
        self.delta_clamp = float(delta_clamp)
        self.feat_proj = nn.Sequential(
            nn.Conv2d(clip_dim, feat_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        in_ch = feat_dim + 1 + (1 if use_u else 0)
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, a0, clip_feat, u_map=None):
        fp = self.feat_proj(clip_feat)
        fp = F.interpolate(fp, size=a0.shape[-2:], mode="bilinear", align_corners=True)
        if self.use_u:
            if u_map is None:
                u_map = torch.zeros_like(a0)
            x = torch.cat([a0, u_map, fp], dim=1)
        else:
            x = torch.cat([a0, fp], dim=1)
        delta = self.refine(x)
        delta = torch.clamp(delta, -self.delta_clamp, self.delta_clamp)
        return (a0 + delta).clamp(0.0, 1.0)


# ================================================================
# Core losses
# ================================================================

def normalize_per_image(a0):
    a0_min = a0.amin(dim=(2, 3), keepdim=True)
    a0_max = a0.amax(dim=(2, 3), keepdim=True)
    a0_range = (a0_max - a0_min).clamp(min=1e-8)
    a0_norm = (a0 - a0_min) / a0_range
    return a0_norm, a0_min, a0_range


def anchor_loss(a_ref, w_n, w_a, lambda_anom=1.5):
    l_n = (w_n * (a_ref ** 2)).sum() / (w_n.sum() + 1e-8)
    l_a = (w_a * ((1.0 - a_ref) ** 2)).sum() / (w_a.sum() + 1e-8)
    return l_n + lambda_anom * l_a


def tv_loss(x):
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return dx + dy


def ranking_loss_from_anchors(a_ref, w_n, w_a, margin=0.02, eps=1e-8):
    s_n = (a_ref * w_n).sum(dim=(1, 2, 3)) / (w_n.sum(dim=(1, 2, 3)) + eps)
    s_a = (a_ref * w_a).sum(dim=(1, 2, 3)) / (w_a.sum(dim=(1, 2, 3)) + eps)
    return F.relu(margin - (s_a - s_n)).mean()


def image_pairwise_rank_loss(s_ref, s_base, margin=0.08, min_gap=0.10, order_temp=0.10):
    # Pairwise order consistency with smooth gradients.
    d_base = s_base[:, None] - s_base[None, :]
    d_ref = s_ref[:, None] - s_ref[None, :]
    keep = d_base.abs() > min_gap
    keep_ratio = keep.float().mean()
    if keep.sum().item() == 0:
        return torch.zeros([], device=s_ref.device), keep_ratio
    sign = torch.sign(d_base)
    z = (sign * d_ref - margin) / max(order_temp, 1e-6)
    loss_mat = F.softplus(-z)
    return loss_mat[keep].mean(), keep_ratio


def normal_invariance_loss(s_ref, s_base):
    # Keep score trend aligned to baseline while allowing absolute shift.
    eps = 1e-6
    sr = (s_ref - s_ref.mean()) / (s_ref.std(unbiased=False) + eps)
    sb = (s_base - s_base.mean()) / (s_base.std(unbiased=False) + eps)
    return F.mse_loss(sr, sb)


def normal_score_calibration_loss(s_ref, s_base, normal_mask, margin=0.0):
    if normal_mask.sum().item() == 0:
        return torch.zeros([], device=s_ref.device)
    drift = s_ref[normal_mask] - s_base[normal_mask] - margin
    return F.relu(drift).mean()


def normal_region_consistency_loss(a_ref, labels_patch, w_n=None, eps=1e-8):
    """
    Keep responses within a normal region coherent.
    """
    bsz = a_ref.shape[0]
    h, w = labels_patch.shape[-2], labels_patch.shape[-1]
    a_patch = F.interpolate(a_ref, size=(h, w), mode="bilinear", align_corners=True)
    a_flat = a_patch.squeeze(1).reshape(bsz, -1)
    lab_flat = labels_patch.reshape(bsz, -1)

    if w_n is None:
        w_flat = torch.ones_like(a_flat)
    else:
        w_patch = F.interpolate(w_n, size=(h, w), mode="bilinear", align_corners=True)
        w_flat = w_patch.squeeze(1).reshape(bsz, -1).clamp(min=0.0)

    k = int(lab_flat.max().item()) + 1
    sum_w = torch.zeros(bsz, k, device=a_ref.device).scatter_add_(1, lab_flat, w_flat).clamp(min=eps)
    sum_wa = torch.zeros(bsz, k, device=a_ref.device).scatter_add_(1, lab_flat, w_flat * a_flat)
    mu = sum_wa / sum_w

    mu_pix = torch.gather(mu, 1, lab_flat)
    var_num = torch.zeros(bsz, k, device=a_ref.device).scatter_add_(
        1, lab_flat, w_flat * (a_flat - mu_pix) ** 2
    )
    return (var_num / sum_w).mean()


def project_refinement(a_in, a_raw, u_in, delta_budget=0.05, u_gate_center=0.5, u_gate_temp=0.15):
    """
    Project raw refinement onto a constrained residual field:
    - zero-bias residual per image
    - bounded residual magnitude (RMS budget)
    - uncertainty-gated application
    """
    eps = 1e-8
    delta_raw = a_raw - a_in
    delta_bias = delta_raw.mean(dim=(2, 3), keepdim=True)
    delta_centered = delta_raw - delta_bias

    rms = torch.sqrt((delta_centered ** 2).mean(dim=(2, 3), keepdim=True) + eps)
    scale = torch.clamp(delta_budget / (rms + eps), max=1.0)
    delta_bounded = delta_centered * scale

    if u_in is None:
        gate = torch.ones_like(delta_bounded)
    else:
        # Conservative policy: higher uncertainty -> smaller update.
        gate = torch.sigmoid((u_gate_center - u_in) / max(u_gate_temp, 1e-6))
        gate = gate.clamp(0.05, 1.0)

    delta_applied = gate * delta_bounded
    a_proj = (a_in + delta_applied).clamp(0.0, 1.0)

    l_bias = delta_bias.abs().mean()
    l_mag = delta_applied.abs().mean()
    return a_proj, delta_applied, l_bias, l_mag


# ================================================================
# UAR-SRA self_refine
# ================================================================

def self_refine(
    a0, clip_features, device,
    u_map=None, region_labels=None,
    n_iter=60, lr=1e-4, gamma=1.5, lambda_anom=1.5,
    lambda_delta=0.2, lambda_tv=0.05, lambda_region=0.05,
    region_tau=0.5, miner_conf=0.55, miner_u0=0.7,
    miner_tau_u=0.3, use_u_input=True,
    lambda_rank=1.0, lambda_img_rank=1.0, lambda_norm_inv=1.0,
    lambda_cal=0.2, cal_margin=0.0, cal_q_normal=0.4,
    rank_margin=0.02, stage1_ratio=0.6,
    delta_budget=0.05, u_gate_center=0.5, u_gate_temp=0.15,
    delta_clamp=0.1, topk_ratio=0.01, pair_min_gap=0.10, order_temp=0.10,
    image_score_mode="topk", image_score_mix_alpha=0.5, global_anchor_alpha=0.5,
    global_quantile_samples=500000,
):
    n_img = a0.shape[0]
    a0_norm, a0_min, a0_range = normalize_per_image(a0)

    net = RefineNet(
        feat_dim=64, clip_dim=clip_features.shape[1], use_u=use_u_input, delta_clamp=delta_clamp
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    a0d = a0_norm.detach().to(device)
    fd = clip_features.detach().to(device)
    ud = u_map.detach().to(device).clamp(0.0, 1.0) if u_map is not None else None
    rd = region_labels.to(device) if region_labels is not None else None

    miner = SoftAnchorMiner(
        confidence_floor=miner_conf, u0=miner_u0, tau_u=miner_tau_u, global_alpha=global_anchor_alpha
    )

    with torch.no_grad():
        s0 = image_score_from_map_torch(
            a0d, mode=image_score_mode, topk_ratio=topk_ratio, mix_alpha=image_score_mix_alpha
        )
        ql_global, qh_global = sampled_global_quantiles(
            a0d, q_low=0.10, q_high=0.97,
            max_samples=global_quantile_samples, seed=111
        )
        ql_global = ql_global.view(1, 1)
        qh_global = qh_global.view(1, 1)
        normal_cut = torch.quantile(s0, cal_q_normal)
        normal_mask_all = s0 <= normal_cut

    best = float("inf")
    patience = 0

    for step in range(n_iter):
        progress = step / max(n_iter - 1, 1)
        bs = min(8, n_img)
        idx = random.sample(range(n_img), bs)

        a_in = a0d[idx]
        f_in = fd[idx]
        u_in = ud[idx] if ud is not None else None
        r_in = rd[idx] if rd is not None else None
        s_base_in = s0[idx]

        a_raw = net(a_in, f_in, u_in)
        a_pred, delta, l_bias, l_mag = project_refinement(
            a_in, a_raw, u_in,
            delta_budget=delta_budget,
            u_gate_center=u_gate_center,
            u_gate_temp=u_gate_temp,
        )

        with torch.no_grad():
            teacher = 0.7 * a_in + 0.3 * a_pred.detach()
            w_n, w_a, valid_ratio = miner.mine_confident(
                teacher, progress, u_map=u_in, ql_global=ql_global, qh_global=qh_global
            )

        la = anchor_loss(a_pred, w_n, w_a, lambda_anom=lambda_anom)
        l_rank = ranking_loss_from_anchors(a_pred, w_n, w_a, margin=rank_margin)
        s_ref = image_score_from_map_torch(
            a_pred, mode=image_score_mode, topk_ratio=topk_ratio, mix_alpha=image_score_mix_alpha
        )
        l_img, pair_keep = image_pairwise_rank_loss(
            s_ref, s_base_in, margin=rank_margin, min_gap=pair_min_gap, order_temp=order_temp
        )
        l_norm_inv = normal_invariance_loss(s_ref, s_base_in)
        l_cal = normal_score_calibration_loss(
            s_ref, s_base_in, normal_mask_all[idx], margin=cal_margin
        )
        lf = F.mse_loss(a_pred, a_in)
        l_tv = tv_loss(delta)
        l_nrc = torch.tensor(0.0, device=device)
        if r_in is not None:
            l_nrc = normal_region_consistency_loss(a_pred, r_in, w_n=w_n)

        # Stage-wise optimization: preserve ranking first, then shape regions.
        stage_prog = max((progress - stage1_ratio) / max(1.0 - stage1_ratio, 1e-6), 0.0)
        delta_sched = 0.30 + 0.70 * progress
        region_sched = stage_prog
        rank_sched = 1.0 - 0.5 * stage_prog
        l_delta = l_bias + 0.5 * l_mag + 0.5 * l_nrc

        l_region = torch.tensor(0.0, device=device)
        if r_in is not None and lambda_region > 0:
            l_region = region_var_loss(a_pred, r_in, u_map=u_in, tau_r=region_tau)

        loss = (
            la
            + lambda_img_rank * l_img
            + (lambda_rank * rank_sched) * l_rank
            + lambda_norm_inv * l_norm_inv
            + lambda_cal * l_cal
            + gamma * lf
            + (lambda_delta * delta_sched) * l_delta
            + lambda_tv * l_tv
            + (lambda_region * region_sched) * l_region
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if loss.item() < best - 1e-6:
            best = loss.item()
            patience = 0
        else:
            patience += 1

        if step % 10 == 0 or step == n_iter - 1:
            with torch.no_grad():
                d_raw = net(a0d[:1], fd[:1], ud[:1] if ud is not None else None)
                a_dbg, d, _, _ = project_refinement(
                    a0d[:1], d_raw, ud[:1] if ud is not None else None,
                    delta_budget=delta_budget,
                    u_gate_center=u_gate_center,
                    u_gate_temp=u_gate_temp,
                )
                d = a_dbg - a0d[:1]
            print(
                f"    {step:3d}: La={la.item():.4f} Limg={l_img.item():.6f} "
                f"Lrank={l_rank.item():.6f} Lninv={l_norm_inv.item():.6f} "
                f"Lcal={l_cal.item():.6f} Lf={lf.item():.6f} "
                f"Lb={l_bias.item():.6f} Lm={l_mag.item():.6f} "
                f"Lnrc={l_nrc.item():.6f} Ld={l_delta.item():.6f} "
                f"sd={delta_sched:.3f} rs={region_sched:.3f} rk={rank_sched:.3f} "
                f"Ltv={l_tv.item():.6f} "
                f"Lreg={l_region.item():.6f} valid={valid_ratio:.3f} "
                f"pair_keep={pair_keep.item():.3f} "
                f"dmax={d.abs().max().item():.4f} dmean={d.mean().item():.6f}"
            )

        if patience >= 25:
            print(f"    Early stop @ {step}")
            break

    net.eval()
    with torch.no_grad():
        parts = []
        for i in range(0, n_img, 8):
            j = min(i + 8, n_img)
            a_raw = net(
                a0d[i:j], fd[i:j],
                ud[i:j] if ud is not None else None,
            )
            a_proj, _, _, _ = project_refinement(
                a0d[i:j], a_raw, ud[i:j] if ud is not None else None,
                delta_budget=delta_budget,
                u_gate_center=u_gate_center,
                u_gate_temp=u_gate_temp,
            )
            parts.append(a_proj.cpu())
        a_refined_norm = torch.cat(parts, dim=0)

    a_refined = a_refined_norm * a0_range + a0_min
    return a_refined


# ================================================================
# Dataset helper — 使用官方 Dataset
# ================================================================

def make_dataloader(data_path, classname, dataset_type, image_size, batch_size, mode="test", meta_path=None):
    """Use official Dataset + official get_transform pipeline."""
    targs = argparse.Namespace(image_size=image_size)
    preprocess, target_transform = get_transform(targs)
    ds = Dataset(
        root=data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=dataset_type,
        mode=mode,
        meta_path=meta_path,
        only_cls=classname,
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return dl


# ================================================================
# Main run
# ================================================================

def run(args):
    seed_everything(args.seed)

    device = torch.device(f"cuda:{args.device}")
    image_size = args.image_size

    all_cls = _CLASSNAMES_mvtec if args.dataset == "mvtec" else _CLASSNAMES_visa
    cats = all_cls if args.class_name.upper() == "ALL" else [args.class_name]

    print("Loading AnomalyCLIP...")
    scorer = AnomalyCLIPScorer(
        device=device,
        image_size=image_size,
        checkpoint=args.checkpoint,
        features_list=args.features_list,
        feature_map_layer=args.feature_map_layer,
        sigma=args.sigma,
    )

    all_results = {}
    for cat in cats:
        print(f"\n{'=' * 60}\n{cat}\n{'=' * 60}")

        try:
            dl = make_dataloader(
                args.data_path, cat, args.dataset, image_size, args.batch_size,
                mode=args.mode, meta_path=args.meta_path
            )
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # --- Collect A0, features, uncertainty ---
        all_a0, all_feat, all_u, all_sp = [], [], [], []
        gt_list, all_mask = [], []

        for items in tqdm(dl, desc=cat):
            img = items["img"].to(device)
            # 官方 dataset 返回 'img_mask' 作为 GT mask
            mask = items.get("img_mask", torch.zeros_like(img[:, :1]))
            is_anomaly = items.get("anomaly", items.get("is_anomaly",
                          (mask.amax(dim=(1, 2, 3)) > 0).long()))

            all_mask.append(mask.cpu())
            gt_list.extend(is_anomaly.cpu().numpy().tolist())

            cls_id = items.get("cls_id", 0)
            if torch.is_tensor(cls_id):
                cls_id = int(cls_id[0].item())
            elif isinstance(cls_id, (list, tuple)):
                cls_id = int(cls_id[0])
            else:
                cls_id = int(cls_id)

            a0, feat, amaps, sp = scorer.score_batch(img, cls_id=cls_id, return_amaps=True)

            u_layer = compute_uncertainty_from_layers(amaps)

            if args.unc_tta:
                # TTA: flip augmentations
                tta_maps = []
                for name, aug, inv in tta_flips(img):
                    if name == "id" or name in args.tta_use.split(","):
                        a_aug, _, _ = scorer.score_batch(aug(img), cls_id=cls_id, return_amaps=False)
                        tta_maps.append(inv(a_aug))
                u_aug = torch.var(torch.stack(tta_maps, dim=0), dim=0, unbiased=False)
            else:
                u_aug = torch.zeros_like(u_layer)

            u = args.alpha_u_layer * u_layer + args.alpha_u_aug * u_aug
            u = normalize_01_per_image(u, q_low=0.05, q_high=0.95)

            all_a0.append(a0.cpu())
            all_feat.append(feat.cpu())
            all_u.append(u.cpu())
            sp = sp.detach().float().cpu().view(-1)
            all_sp.extend(sp.tolist())

        a0 = torch.cat(all_a0).detach()
        feats = torch.cat(all_feat).detach()
        u_map = torch.cat(all_u).detach()
        pr_sp_base = np.asarray(all_sp, dtype=np.float32)

        gt_masks = torch.cat(all_mask)
        # gt mask: threshold at 0.5
        gt_px = (gt_masks.squeeze(1).numpy() > 0.5).astype(np.int32)
        gt_sp = np.array(gt_list, dtype=np.int32)

        n_img = a0.shape[0]
        if args.use_clip_sp_for_base and pr_sp_base.shape[0] != n_img:
            print(f"  [warn] clip sp len mismatch: pr={pr_sp_base.shape[0]} vs n_img={n_img}; fallback to map score for base image metrics")
            args.use_clip_sp_for_base = False
        a0_sq = a0.squeeze(1)

        print(f"  N={n_img}, anom={gt_sp.sum()}, gt_px_sum={gt_px.sum()}")
        print(f"  A0 range: [{a0.min():.6f}, {a0.max():.6f}]")
        print(f"  U  range: [{u_map.min():.6f}, {u_map.max():.6f}], mean={u_map.mean():.6f}")

        # --- Baseline metrics ---
        pr_px_base = a0_sq.detach().numpy()
        pr_sp_base_eval = pr_sp_base if args.use_clip_sp_for_base else None
        m_base = compute_metrics(
            gt_sp, gt_px, pr_px_base, pr_sp=pr_sp_base_eval, sigma=args.sigma,
            image_score_mode=args.image_score_mode,
            image_score_topk_ratio=args.topk_ratio,
            image_score_mix_alpha=args.image_score_mix_alpha,
        )
        print(
            f"  [base] image_auroc={m_base['image_auroc']*100:.1f} "
            f"image_ap={m_base['image_ap']*100:.1f} "
            f"pixel_auroc={m_base['pixel_auroc']*100:.1f} "
            f"pixel_aupro={m_base['pixel_aupro']*100:.1f}"
        )

        # --- Build region labels ---
        if args.region_method == "none":
            region_labels = None
        elif args.region_method == "grid":
            _, _, rh, rw = feats.shape
            base_lab = build_grid_regions(rh, rw, grid=args.grid_regions)
            region_labels = base_lab.unsqueeze(0).repeat(feats.shape[0], 1, 1)
        elif args.region_method == "kmeans_clip":
            region_labels = build_regions_from_clip_feat(
                feats, n_regions=args.n_regions,
                n_iter=args.kmeans_iter, device=str(device),
            )
        else:
            raise ValueError(f"Unknown region_method: {args.region_method}")

        # --- UAR-SRA ---
        print(
            f"  UAR-SRA: n_iter={args.n_iter} lr={args.lr} gamma={args.gamma} "
            f"lam_anom={args.lambda_anom} lam_delta={args.lambda_delta} "
            f"lam_tv={args.lambda_tv} lam_region={args.lambda_region} "
            f"lam_rank={args.lambda_rank} lam_cal={args.lambda_cal} stage1={args.stage1_ratio} "
            f"dbudget={args.delta_budget} "
            f"score={args.image_score_mode} "
            f"region={args.region_method} u0={args.u0} tau_u={args.tau_u} "
            f"use_u_input={args.use_u_input}"
        )
        t0 = time.time()
        if args.refiner == 'ulgp':
            from ulgp import ULGPPlugin, ULGPConfig
            cfg = ULGPConfig(
                mode=args.ulgp_mode,
                candidate_ratio=args.ulgp_candidate_ratio,
                dilation_radius=args.ulgp_dilation_radius,
                k=args.ulgp_k,
                num_steps=args.ulgp_steps,
                alpha=args.ulgp_alpha,
                beta=args.ulgp_beta,
                tau_u=args.ulgp_tau_u,
                clamp_delta=args.ulgp_clamp_delta,
                fusion_lambda=args.ulgp_fusion_lambda,
            )
            plugin = ULGPPlugin(config=cfg)
            a_sra, ulgp_info = plugin.refine(
                anomaly_map=a0,
                feature_map=feats,
                uncertainty_map=u_map,
                image=None,
                return_info=True,
            )
            print(
                f"  ULGP: mode={ulgp_info['mode']} graph_space={ulgp_info['graph_space']} "
                f"nodes={ulgp_info['num_nodes']} edges={ulgp_info['num_edges']} "
                f"mean_upd={ulgp_info['mean_update']:.6f} gate_mean={ulgp_info['gate_mean']:.4f} "
                f"clamped={ulgp_info['clamped_fraction']:.4f}"
            )
        else:
            a_sra = self_refine(
                a0, feats, device,
                u_map=u_map, region_labels=region_labels,
                n_iter=args.n_iter, lr=args.lr, gamma=args.gamma,
                lambda_anom=args.lambda_anom, lambda_delta=args.lambda_delta,
                lambda_tv=args.lambda_tv, lambda_region=args.lambda_region,
                region_tau=args.region_tau, miner_conf=args.confidence_floor,
                miner_u0=args.u0, miner_tau_u=args.tau_u,
                use_u_input=args.use_u_input,
                lambda_rank=args.lambda_rank, lambda_img_rank=args.lambda_img_rank,
                lambda_norm_inv=args.lambda_norm_inv, rank_margin=args.rank_margin,
                lambda_cal=args.lambda_cal, cal_margin=args.cal_margin, cal_q_normal=args.cal_q_normal,
                stage1_ratio=args.stage1_ratio, delta_budget=args.delta_budget,
                u_gate_center=args.u_gate_center, u_gate_temp=args.u_gate_temp,
                delta_clamp=args.delta_clamp, topk_ratio=args.topk_ratio,
                pair_min_gap=args.pair_min_gap, order_temp=args.order_temp,
                image_score_mode=args.image_score_mode, image_score_mix_alpha=args.image_score_mix_alpha,
                global_anchor_alpha=args.global_anchor_alpha,
                global_quantile_samples=args.global_quantile_samples,
            )
        print(f"  UAR-SRA: {time.time() - t0:.1f}s")

        a_sra_sq = a_sra.squeeze(1)
        pr_px_sra = a_sra_sq.detach().numpy()

        pr_sp_sra = pr_sp_base if args.reuse_base_image_score else None
        m_sra = compute_metrics(
            gt_sp, gt_px, pr_px_sra, pr_sp=pr_sp_sra, sigma=args.sigma,
            image_score_mode=args.image_score_mode,
            image_score_topk_ratio=args.topk_ratio,
            image_score_mix_alpha=args.image_score_mix_alpha,
        )
        print(
            f"  [uar]  image_auroc={m_sra['image_auroc']*100:.1f} "
            f"image_ap={m_sra['image_ap']*100:.1f} "
            f"pixel_auroc={m_sra['pixel_auroc']*100:.1f} "
            f"pixel_aupro={m_sra['pixel_aupro']*100:.1f}"
        )

        d = (a_sra - a0).detach()
        print(
            f"  delta: mean={d.mean():.6f} std={d.std():.6f} "
            f"max={d.abs().max():.6f}"
        )

        all_results[cat] = {"baseline": m_base, "sra": m_sra}

    # --- Summary ---
    if len(all_results) >= 2:
        cats_done = list(all_results.keys())
        for mn in ["pixel_auroc", "pixel_aupro", "image_auroc", "image_ap"]:
            print(f"\n{'=' * 60}\n{mn}\n{'=' * 60}")
            print(f"{'Cat':<14}{'base':>10}{'uar':>10}{'D':>8}")
            for c in cats_done:
                b = all_results[c]["baseline"][mn]
                a = all_results[c]["sra"][mn]
                print(f"{c:<14}{b*100:>9.1f}%{a*100:>9.1f}%{(a-b)*100:>+7.2f}")
            bm = np.mean([all_results[c]["baseline"][mn] for c in cats_done])
            am = np.mean([all_results[c]["sra"][mn] for c in cats_done])
            print(f"{'MEAN':<14}{bm*100:>9.1f}%{am*100:>9.1f}%{(am-bm)*100:>+7.2f}")


def parse_args():
    p = argparse.ArgumentParser(description="UAR-SRA for AnomalyCLIP", conflict_handler="resolve")
    p.add_argument("--class_name", type=str, default="ALL")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    p.add_argument("--data_path", type=str, default="")
    p.add_argument("--meta_path", type=str, default=None)
    p.add_argument("--mode", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--image_size", type=int, default=518)

    # 对齐官方 test.py 的参数
    p.add_argument("--features_list", type=int, nargs="+", default=[6,12,18,24])
    p.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--depth", type=int, default=9)
    p.add_argument("--n_ctx", type=int, default=12)
    p.add_argument("--t_n_ctx", type=int, default=4)

    # SRA core
    p.add_argument("--n_iter", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--lambda_anom", type=float, default=1.5)
    p.add_argument("--lambda_delta", type=float, default=0.2)
    p.add_argument("--lambda_tv", type=float, default=0.05)
    p.add_argument("--lambda_rank", type=float, default=1.0)
    p.add_argument("--lambda_img_rank", type=float, default=1.0)
    p.add_argument("--lambda_norm_inv", type=float, default=1.0)
    p.add_argument("--lambda_cal", type=float, default=0.2)
    p.add_argument("--rank_margin", type=float, default=0.02)
    p.add_argument("--stage1_ratio", type=float, default=0.6)
    p.add_argument("--delta_budget", type=float, default=0.05)
    p.add_argument("--delta_clamp", type=float, default=0.1)
    p.add_argument("--u_gate_center", type=float, default=0.5)
    p.add_argument("--u_gate_temp", type=float, default=0.15)
    p.add_argument("--topk_ratio", type=float, default=0.01)
    p.add_argument("--pair_min_gap", type=float, default=0.10)
    p.add_argument("--order_temp", type=float, default=0.10)
    p.add_argument("--cal_margin", type=float, default=0.0)
    p.add_argument("--cal_q_normal", type=float, default=0.4)
    p.add_argument("--image_score_mode", type=str, default="topk", choices=["max", "topk", "mix"])
    p.add_argument("--image_score_mix_alpha", type=float, default=0.5)
    p.add_argument("--global_anchor_alpha", type=float, default=0.5)
    p.add_argument("--global_quantile_samples", type=int, default=500000)
    p.add_argument("--reuse_base_image_score", action="store_true")
    p.add_argument("--use_clip_sp_for_base", action="store_true")

    # Uncertainty
    p.add_argument("--unc_tta", action="store_true")
    p.add_argument("--tta_use", type=str, default="h,v")
    p.add_argument("--alpha_u_layer", type=float, default=1.0)
    p.add_argument("--alpha_u_aug", type=float, default=1.0)

    # Uncertainty-aware anchor mining
    p.add_argument("--confidence_floor", type=float, default=0.55)
    p.add_argument("--u0", type=float, default=0.7)
    p.add_argument("--tau_u", type=float, default=0.3)

    # Region
    p.add_argument("--region_method", type=str, default="kmeans_clip",
                   choices=["none", "grid", "kmeans_clip"])
    p.add_argument("--n_regions", type=int, default=64)
    p.add_argument("--kmeans_iter", type=int, default=10)
    p.add_argument("--grid_regions", type=int, default=8)
    p.add_argument("--lambda_region", type=float, default=0.05)
    p.add_argument("--region_tau", type=float, default=0.5)

    # RefineNet
    p.set_defaults(use_u_input=True)
    p.add_argument("--use_u_input", dest="use_u_input", action="store_true")
    p.add_argument("--no_use_u_input", dest="use_u_input", action="store_false")

    # Misc
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--sigma", type=float, default=4.0)

    # Refiner selector
    p.add_argument("--refiner", type=str, default="ulgp", choices=["ulgp", "sra"])

    # ULGP params
    p.add_argument("--ulgp_mode", type=str, default="feature-lite", choices=["auto", "full", "feature-lite", "map-only"])
    p.add_argument("--ulgp_candidate_ratio", type=float, default=0.05)
    p.add_argument("--ulgp_dilation_radius", type=int, default=1)
    p.add_argument("--ulgp_k", type=int, default=8)
    p.add_argument("--ulgp_steps", type=int, default=3)
    p.add_argument("--ulgp_alpha", type=float, default=0.5)
    p.add_argument("--ulgp_beta", type=float, default=0.1)
    p.add_argument("--ulgp_tau_u", type=float, default=0.5)
    p.add_argument("--ulgp_clamp_delta", type=float, default=0.05)
    p.add_argument("--ulgp_fusion_lambda", type=float, default=0.7)

    args = p.parse_args()

    if not args.data_path:
        args.data_path = (
            "/home/dct/unified_model/data/mvtec"
            if args.dataset == "mvtec"
            else "/home/dct/unified_model/data/VisA_20220922"
        )

    if not args.checkpoint:
        if args.dataset == "mvtec":
            args.checkpoint = os.path.join(
                os.path.dirname(__file__), "..",
                "checkpoints/9_12_4_multiscale_mvtec1/epoch_15.pth",
            )
        else:
            args.checkpoint = os.path.join(
                os.path.dirname(__file__), "..",
                "checkpoints/9_12_4_multiscale_visa/epoch_15.pth",
            )

    return args


if __name__ == "__main__":
    run(parse_args())
