#!/usr/bin/env python
"""Projection utilities for 2D->3D alignment (ported from ESAM/OneFormer3D).

This module intentionally stays small and self-contained so it can be reused
by multiple 2D backbones (DINO/GroundingDINO/CLIP).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

# ScanNet default intrinsics for the canonical 640x480 images.
SCANET_INTRINSICS = (577.870605, 577.870605, 319.5, 239.5)
BASE_IMAGE_SIZE = (480.0, 640.0)
MIN_DEPTH = 0.3


def _scale_intrinsics(
    standard_intrinsics: Tuple[float, float, float, float],
    feat_hw: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    fx0, fy0, cx0, cy0 = standard_intrinsics
    h_feat, w_feat = feat_hw
    base_h, base_w = BASE_IMAGE_SIZE
    scale_w = float(w_feat) / float(base_w)
    scale_h = float(h_feat) / float(base_h)
    fx_feat = fx0 * scale_w
    fy_feat = fy0 * scale_h
    cx_feat = cx0 * scale_w
    cy_feat = cy0 * scale_h
    return fx_feat, fy_feat, cx_feat, cy_feat


def pixels_to_grid(
    uv_feat: torch.Tensor,
    feat_hw: Tuple[int, int],
    *,
    align_corners: bool = True,
) -> torch.Tensor:
    h, w = feat_hw
    u = uv_feat[:, 0]
    v = uv_feat[:, 1]
    if align_corners:
        x_norm = 2.0 * u / max(float(w - 1), 1.0) - 1.0
        y_norm = 2.0 * v / max(float(h - 1), 1.0) - 1.0
    else:
        x_norm = 2.0 * (u + 0.5) / float(w) - 1.0
        y_norm = 2.0 * (v + 0.5) / float(h) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    return grid


def sample_img_feat(
    feat_map: torch.Tensor,
    uv_feat: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    align_corners: bool = True,
) -> torch.Tensor:
    """Sample point-wise features from a (1,C,H,W) feature map.

    Returns:
        (N,C) tensor. Invalid points are zero-filled.
    """
    assert feat_map.dim() == 4 and feat_map.size(0) == 1
    h, w = int(feat_map.shape[-2]), int(feat_map.shape[-1])
    idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return feat_map.new_zeros((uv_feat.size(0), feat_map.size(1)))
    uv_valid = uv_feat[idx]
    grid = pixels_to_grid(uv_valid, (h, w), align_corners=align_corners)
    if grid.dtype != feat_map.dtype:
        grid = grid.to(feat_map.dtype)
    sampled = F.grid_sample(
        feat_map, grid, mode="bilinear", align_corners=align_corners
    ).squeeze(3).squeeze(0).T
    out = feat_map.new_zeros((uv_feat.size(0), feat_map.size(1)))
    out[idx] = sampled
    return out


def scale_uv_img_to_feat(
    uv_img: torch.Tensor,
    *,
    img_hw: Tuple[int, int],
    feat_hw: Tuple[int, int],
    align_corners: bool = False,
) -> torch.Tensor:
    """Scale image pixel coords to feature-map pixel coords.

    This is used when intrinsics are defined in the resized image space
    (H_img,W_img) but we want to sample / validate on a feature map of
    (H_feat,W_feat).

    For `align_corners=False` (recommended), we use the half-pixel rule:
        u_feat = (u_img + 0.5) * (W_feat / W_img) - 0.5
    For `align_corners=True`, use:
        u_feat = u_img * ((W_feat-1) / (W_img-1))
    """
    h_img, w_img = int(img_hw[0]), int(img_hw[1])
    h_feat, w_feat = int(feat_hw[0]), int(feat_hw[1])
    u = uv_img[:, 0]
    v = uv_img[:, 1]
    if align_corners:
        sw = float(w_feat - 1) / max(float(w_img - 1), 1.0)
        sh = float(h_feat - 1) / max(float(h_img - 1), 1.0)
        u_feat = u * sw
        v_feat = v * sh
    else:
        sw = float(w_feat) / max(float(w_img), 1.0)
        sh = float(h_feat) / max(float(h_img), 1.0)
        u_feat = (u + 0.5) * sw - 0.5
        v_feat = (v + 0.5) * sh - 0.5
    return torch.stack([u_feat, v_feat], dim=-1)


def project_points_to_uv(
    xyz_cam: torch.Tensor,
    *,
    feat_hw: Tuple[int, int],
    max_depth: float,
    standard_intrinsics: Tuple[float, float, float, float] = SCANET_INTRINSICS,
    already_scaled: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D points in camera coordinates to 2D feature grid coordinates.

    Args:
        xyz_cam: (N,3) points in camera coordinates.
        feat_hw: (H_feat, W_feat) of the target grid.
        max_depth: maximum valid depth.
        standard_intrinsics: intrinsics of the base image or already-scaled intrinsics.
        already_scaled: if True, do NOT rescale intrinsics inside this function.

    Returns:
        uv_feat: (N,2) float pixel coords in feature grid
        valid: (N,) bool mask
    """
    if already_scaled:
        fx_feat, fy_feat, cx_feat, cy_feat = standard_intrinsics
    else:
        fx_feat, fy_feat, cx_feat, cy_feat = _scale_intrinsics(
            standard_intrinsics, feat_hw
        )
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    valid_z = (z > MIN_DEPTH) & (z < max_depth)

    ratio_x = torch.zeros_like(x)
    ratio_y = torch.zeros_like(y)
    valid_ratio_mask = valid_z & (torch.abs(z) > torch.finfo(z.dtype).eps)
    if valid_ratio_mask.any():
        ratio_x[valid_ratio_mask] = x[valid_ratio_mask] / z[valid_ratio_mask]
        ratio_y[valid_ratio_mask] = y[valid_ratio_mask] / z[valid_ratio_mask]

    u_feat = fx_feat * ratio_x + cx_feat
    v_feat = fy_feat * ratio_y + cy_feat
    u_feat = torch.where(valid_z, u_feat, torch.full_like(u_feat, -1.0))
    v_feat = torch.where(valid_z, v_feat, torch.full_like(v_feat, -1.0))

    w_feat = float(feat_hw[1])
    h_feat = float(feat_hw[0])
    valid_u = (u_feat >= 0) & (u_feat < w_feat)
    valid_v = (v_feat >= 0) & (v_feat < h_feat)
    valid = valid_z & valid_u & valid_v

    uv_feat = torch.stack([u_feat, v_feat], dim=-1)
    return uv_feat, valid
