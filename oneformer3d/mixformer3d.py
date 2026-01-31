import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_scatter import scatter_mean, scatter
import MinkowskiEngine as ME
import pointops
import pdb, time
from functools import partial
from contextlib import nullcontext
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.models import Base3DDetector
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type
from mmengine.structures import InstanceData
from .mask_matrix_nms import mask_matrix_nms
from .oneformer3d import ScanNetOneFormer3DMixin
from .instance_merge import ins_merge_mat, ins_cat, ins_merge, OnlineMerge, GTMerge, DQ_Track_OnlineMerge
import numpy as np
from .img_backbone import point_sample, apply_3d_transformation
import os
from PIL import Image
from .projection_utils import MIN_DEPTH, scale_uv_img_to_feat, project_points_to_uv, sample_img_feat
# Optional disk cache for GDINO outputs (srcs/hs_last/boxes/scores).
from .gdino_cache import load_gdino_cache_batched, load_gdino_cache_single
# Sparse FPN builder (ESAM-style): point-wise 2D feats -> [s1,s2,s4,s8,s16]
from .dino_sparse_fpn import build_sparse_fpn
# Added
from mmdet3d.registry import TASK_UTILS
from mmengine.model import BaseModule
from oneformer3d.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou
from oneformer3d.util import box_ops, checkpoint
from oneformer3d.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .dq_utils import match_for_indices, QueryInteractionX, build_pairwise_mask, cluster_with_threshold, analyze_masked_stats, find_optimal_threshold, calc_mean_grouped, frozen_inference, MergeFusion, sigmoid_focal_loss, cluster_complete_link, evaluate_clustering_pairwise,cluster_with_per_class_threshold
from .dq_utils import FFN as DQ_FFN
from .dq_utils import replace_bn_with_ln as replace_bn
from .motr_utils import SelfAttention as MOTR_SelfAttention
from .motr_utils import FFN as MOTR_FFN
from scipy.optimize import linear_sum_assignment
from mmdet3d.structures import AxisAlignedBboxOverlaps3D
from easydict import EasyDict

@MODELS.register_module()
class ScanNet200MixFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 backbone=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 criterion=None,
                 gdino_backbone=None,
                 gdino_point_fusion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 use_one2many=False,
                 criterion_one2many=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.pool = MODELS.build(pool)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_one2many = use_one2many
        if self.use_one2many:
            self.one2many_loss_weight = 0.5
            self.criterion_one2many = MODELS.build(criterion_one2many[0])

        # --- Optional: GroundingDINO side-channel modules (offline SV compatible) ---
        # Fully gated by config; must not change baseline behavior when disabled.
        self._gdino_backbone_cfg = gdino_backbone or {}
        self.gdino_point_fusion_cfg = gdino_point_fusion or {}
        self._gdino_backbone = None
        self._gdino_point_proj = None
        try:
            if isinstance(self.gdino_point_fusion_cfg, dict) and bool(self.gdino_point_fusion_cfg.get("enable", False)):
                in_dim = int(self.gdino_point_fusion_cfg.get("in_dim", 256))
                out_dim = int(self.gdino_point_fusion_cfg.get("out_dim", in_dim))
                proj_type = str(self.gdino_point_fusion_cfg.get("proj_type", "identity")).lower()
                if proj_type == "identity" and in_dim == out_dim:
                    self._gdino_point_proj = nn.Identity()
                else:
                    self._gdino_point_proj = nn.Linear(in_dim, out_dim, bias=False)
        except Exception:
            self._gdino_point_proj = None

        # Side-channel: SP positions in the wo-elastic coordinate space for DACA-2D.
        self._last_sp_pos_wo_elastic_list = None

    # -------------------------------------------------------------------------
    # GDINO helpers (offline SV).
    # -------------------------------------------------------------------------
    def _get_gdino_backbone(self, cfg=None):
        """Lazily build GroundingDINO backbone for point-fusion / DACA-2D."""
        bb = getattr(self, "_gdino_backbone", None)
        if bb is not None:
            return bb
        bb_cfg = None
        if isinstance(cfg, dict):
            bb_cfg = cfg.get("backbone", None)
        if not isinstance(bb_cfg, dict) and isinstance(getattr(self, "_gdino_backbone_cfg", None), dict):
            bb_cfg = getattr(self, "_gdino_backbone_cfg")
        if isinstance(bb_cfg, dict) and len(bb_cfg) > 0:
            bb = MODELS.build(bb_cfg)
        else:
            from .gdino_backbone import GroundingDINOBackbone
            bb = GroundingDINOBackbone()
        self._gdino_backbone = bb
        return bb

    def _get_gdino_daca2d_cfg(self, *, is_train: bool) -> dict:
        """Get DACA-2D config from train/test cfg with safe fallback."""
        cfg = {}
        try:
            base = (self.train_cfg or {}) if is_train else (self.test_cfg or {})
            cfg = base.get("gdino_daca2d", {}) or {}
        except Exception:
            cfg = {}
        if not cfg:
            try:
                cfg = (self.test_cfg or {}).get("gdino_daca2d", {}) or {}
            except Exception:
                cfg = {}
        return cfg if isinstance(cfg, dict) else {}

    @staticmethod
    def _unwrap_cam_and_img(meta: dict, frame_i: int = 0):
        cam_info = meta.get("cam_info", None)
        img_paths = meta.get("img_paths", None)
        if isinstance(cam_info, dict):
            cam_list = [cam_info]
        elif isinstance(cam_info, list) and len(cam_info) > 0:
            cam_list = cam_info
        else:
            return None, None
        if isinstance(img_paths, list) and len(img_paths) > 0:
            img_path = img_paths[frame_i] if frame_i < len(img_paths) else img_paths[0]
        elif isinstance(img_paths, str):
            img_path = img_paths
        else:
            img_path = None
        cam = cam_list[frame_i] if frame_i < len(cam_list) else cam_list[0]
        if not isinstance(cam, dict) or not isinstance(img_path, str):
            return None, None
        return cam, img_path

    @staticmethod
    def _to_tensor_img(img: Image.Image, *, device: torch.device) -> torch.Tensor:
        arr = np.asarray(img).copy()  # ensure writable for torch.from_numpy
        img_t = torch.from_numpy(arr).to(device=device).float() / 255.0
        if img_t.dim() == 3 and img_t.shape[-1] == 3:
            img_t = img_t.permute(2, 0, 1).contiguous()
        return img_t

    @staticmethod
    def _log_mmengine(msg: str):
        try:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            if logger is not None:
                logger.info(msg)
                return
        except Exception:
            pass
        print(msg)

    def _run_gdino_point_fusion_for_frame(self, batch_inputs_dict, batch_data_samples, frame_i: int = 0):
        """Point-level early fusion: sample image-level GDINO srcs for each point."""
        cfg = self.gdino_point_fusion_cfg or {}
        if not (isinstance(cfg, dict) and bool(cfg.get("enable", False))):
            return None, None
        if self._gdino_point_proj is None:
            return None, {"skipped": "no_proj", "frame": int(frame_i)}

        feat_levels = cfg.get("feat_levels", None)
        if feat_levels is None:
            feat_levels = [int(cfg.get("feat_level", 0))]
        feat_levels = [int(l) for l in (feat_levels if isinstance(feat_levels, (list, tuple)) else [feat_levels])]

        align_corners = bool(cfg.get("align_corners", False))
        max_depth = float(cfg.get("max_depth", 10.0))
        strict = bool(cfg.get("strict", False))
        strict_thr = float(cfg.get("strict_valid_ratio", 0.95))

        pts_raw_in = batch_inputs_dict.get("points_raw", None)
        pts_aug_in = batch_inputs_dict.get("points", None)
        if pts_aug_in is None:
            return None, {"skipped": "no_points", "frame": int(frame_i)}
        B = len(batch_data_samples)
        device = pts_aug_in[0].device

        cams, img_paths = [], []
        for b in range(B):
            meta_b = getattr(batch_data_samples[b], "metainfo", None)
            if callable(meta_b):
                meta_b = meta_b()
            # Some custom Pack3DDetInputs_ variants store 2D-3D metadata in
            # `img_metas` instead of `metainfo()`. Prefer `img_metas` when
            # required keys are missing, otherwise GDINO will treat the sample
            # as invalid and skip the whole batch.
            if (not isinstance(meta_b, dict)) or ('cam_info' not in meta_b and 'img_paths' not in meta_b):
                meta_b = getattr(batch_data_samples[b], "img_metas", {}) or (meta_b if isinstance(meta_b, dict) else {})
            cam, img_path = self._unwrap_cam_and_img(meta_b, frame_i=int(frame_i))
            if cam is None or img_path is None:
                cams.append(None)
                img_paths.append(None)
                continue
            cams.append(cam)
            img_paths.append(img_path)

        srcs = None
        cache_dir = cfg.get("cache_dir", None)
        if cache_dir is None and isinstance(cfg.get("cache", None), dict):
            cache_dir = cfg.get("cache", {}).get("dir", None) or cfg.get("cache", {}).get("cache_dir", None)
        if cache_dir is None:
            cache_dir = os.environ.get("GDINO_CACHE_DIR", None)

        if cache_dir is not None and all(isinstance(p, str) for p in img_paths) and all(isinstance(c, dict) for c in cams):
            # Require consistent target size across the batch.
            hw0 = cams[0].get("img_size_gdino", None)
            if torch.is_tensor(hw0) and hw0.numel() == 2:
                target_hw = (int(hw0.reshape(-1)[0].item()), int(hw0.reshape(-1)[1].item()))
                bb_cfg = None
                if isinstance(cfg.get("gdino", None), dict):
                    bb_cfg = cfg["gdino"].get("backbone", None)
                if not isinstance(bb_cfg, dict):
                    bb_cfg = getattr(self, "_gdino_backbone_cfg", None)
                cached = load_gdino_cache_batched(
                    cache_dir,
                    img_paths=[str(p) for p in img_paths],
                    target_hw=target_hw,
                    bb_cfg=bb_cfg if isinstance(bb_cfg, dict) else None,
                    mode="full",
                    device=device,
                    dtype=torch.float32,
                )
                if isinstance(cached, dict):
                    srcs = cached.get("srcs", None)
                if not isinstance(srcs, list) or len(srcs) == 0:
                    cached = load_gdino_cache_batched(
                        cache_dir,
                        img_paths=[str(p) for p in img_paths],
                        target_hw=target_hw,
                        bb_cfg=bb_cfg if isinstance(bb_cfg, dict) else None,
                        mode="backbone",
                        device=device,
                        dtype=torch.float32,
                    )
                    if isinstance(cached, dict):
                        srcs = cached.get("srcs", None)

        if not isinstance(srcs, list) or len(srcs) == 0:
            imgs = []
            for b in range(B):
                cam = cams[b]
                img_path = img_paths[b]
                if cam is None or img_path is None:
                    imgs.append(None)
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_size = cam.get("img_size_gdino", None)
                    if torch.is_tensor(img_size):
                        h1, w1 = int(img_size[0].item()), int(img_size[1].item())
                        if (img.height, img.width) != (h1, w1):
                            img = img.resize((w1, h1), resample=Image.BILINEAR)
                    imgs.append(img)
                except Exception:
                    imgs.append(None)

            if any(im is None for im in imgs):
                return None, {"skipped": "img_load_failed", "frame": int(frame_i)}

            img_t = torch.stack([self._to_tensor_img(im, device=device) for im in imgs], dim=0)
            gdino = self._get_gdino_backbone(cfg.get("gdino", None))
            out = gdino(img_t, backbone_only=bool(cfg.get("backbone_only", True)))
            srcs = out.get("srcs", None)
        if not isinstance(srcs, list) or len(srcs) == 0:
            return None, {"skipped": "no_srcs", "frame": int(frame_i)}

        out_feats, valid_ratios, best_modes = [], [], []
        for b in range(B):
            cam = cams[b]
            if cam is None:
                out_feats.append(None)
                valid_ratios.append(0.0)
                best_modes.append("na")
                continue
            intr = cam.get("intrinsics", None)
            if isinstance(intr, (list, tuple)) and len(intr) == 1 and torch.is_tensor(intr[0]):
                intr_t = intr[0].to(device=device, dtype=torch.float32).reshape(-1)[:4]
            elif torch.is_tensor(intr):
                intr_t = intr.to(device=device, dtype=torch.float32).reshape(-1)[:4]
            else:
                intr_t = torch.as_tensor(intr, device=device, dtype=torch.float32).reshape(-1)[:4]
            fx, fy, cx, cy = [float(v) for v in intr_t.tolist()]
            pose = cam.get("pose", cam.get("extrinsics", None))
            if torch.is_tensor(pose):
                pose_t = pose.to(device=device, dtype=torch.float32).reshape(4, 4)
            else:
                pose_t = torch.as_tensor(pose, device=device, dtype=torch.float32).reshape(4, 4)
            img_hw = cam.get("img_size_gdino", None)
            if torch.is_tensor(img_hw):
                h_img, w_img = int(img_hw[0].item()), int(img_hw[1].item())
            else:
                h_img, w_img = int(img_t.shape[-2]), int(img_t.shape[-1])

            pts_aug = pts_aug_in[b]
            xyz_aug = pts_aug[:, :3]
            if isinstance(pts_raw_in, (list, tuple)) and len(pts_raw_in) == B:
                xyz_raw = pts_raw_in[b][:, :3]
            else:
                xyz_raw = xyz_aug

            def _ratio_for_mode(mode: str) -> float:
                if mode == "inv":
                    mat = torch.linalg.inv(pose_t)
                elif mode == "direct":
                    mat = pose_t
                else:
                    mat = torch.eye(4, device=device, dtype=pose_t.dtype)
                xyz1 = torch.cat([xyz_raw, torch.ones((xyz_raw.shape[0], 1), device=device, dtype=xyz_raw.dtype)], dim=1)
                xyz_cam = (xyz1 @ mat.T)[:, :3]
                x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
                valid_z = (z > float(MIN_DEPTH)) & (z < max_depth) & (torch.abs(z) > torch.finfo(z.dtype).eps)
                ratio_x = torch.zeros_like(x)
                ratio_y = torch.zeros_like(y)
                if valid_z.any():
                    ratio_x[valid_z] = x[valid_z] / z[valid_z]
                    ratio_y[valid_z] = y[valid_z] / z[valid_z]
                u_img = fx * ratio_x + cx
                v_img = fy * ratio_y + cy
                lv0 = max(0, min(feat_levels[0], len(srcs) - 1))
                feat_map0 = srcs[lv0][b:b+1]
                feat_h0, feat_w0 = int(feat_map0.shape[-2]), int(feat_map0.shape[-1])
                uv_feat0 = scale_uv_img_to_feat(
                    torch.stack([u_img, v_img], dim=-1),
                    img_hw=(h_img, w_img),
                    feat_hw=(feat_h0, feat_w0),
                    align_corners=align_corners,
                )
                valid_u = (uv_feat0[:, 0] >= 0) & (uv_feat0[:, 0] < float(feat_w0))
                valid_v = (uv_feat0[:, 1] >= 0) & (uv_feat0[:, 1] < float(feat_h0))
                valid = valid_z & valid_u & valid_v
                return float(valid.float().mean().item())

            best_mode = "inv"
            best_ratio = _ratio_for_mode("inv")
            for m in ("direct", "identity"):
                r = _ratio_for_mode(m)
                if r > best_ratio:
                    best_ratio, best_mode = r, m
            best_modes.append(best_mode)

            if best_mode == "inv":
                mat = torch.linalg.inv(pose_t)
            elif best_mode == "direct":
                mat = pose_t
            else:
                mat = torch.eye(4, device=device, dtype=pose_t.dtype)
            xyz1 = torch.cat([xyz_raw, torch.ones((xyz_raw.shape[0], 1), device=device, dtype=xyz_raw.dtype)], dim=1)
            xyz_cam = (xyz1 @ mat.T)[:, :3]
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            valid_z = (z > float(MIN_DEPTH)) & (z < max_depth) & (torch.abs(z) > torch.finfo(z.dtype).eps)
            ratio_x = torch.zeros_like(x)
            ratio_y = torch.zeros_like(y)
            if valid_z.any():
                ratio_x[valid_z] = x[valid_z] / z[valid_z]
                ratio_y[valid_z] = y[valid_z] / z[valid_z]
            u_img = fx * ratio_x + cx
            v_img = fy * ratio_y + cy
            uv_img = torch.stack([u_img, v_img], dim=-1)

            feats_lv = []
            valid_any = None
            for lv in feat_levels:
                lv = max(0, min(int(lv), len(srcs) - 1))
                feat_map = srcs[lv][b:b+1]
                feat_h, feat_w = int(feat_map.shape[-2]), int(feat_map.shape[-1])
                uv_feat = scale_uv_img_to_feat(
                    uv_img, img_hw=(h_img, w_img), feat_hw=(feat_h, feat_w), align_corners=align_corners
                )
                valid_u = (uv_feat[:, 0] >= 0) & (uv_feat[:, 0] < float(feat_w))
                valid_v = (uv_feat[:, 1] >= 0) & (uv_feat[:, 1] < float(feat_h))
                valid = valid_z & valid_u & valid_v
                valid_any = valid if valid_any is None else (valid_any | valid)
                feats_lv.append(sample_img_feat(feat_map, uv_feat, valid, align_corners=align_corners))
            feat_pts = torch.stack(feats_lv, dim=0).mean(dim=0) if len(feats_lv) > 1 else feats_lv[0]
            feat_pts = self._gdino_point_proj(feat_pts)
            out_feats.append(feat_pts)
            vr = float(valid_any.float().mean().item()) if valid_any is not None else 0.0
            valid_ratios.append(vr)
            if strict and vr < strict_thr:
                raise RuntimeError(
                    f"[GDINO][point_fusion][strict] valid_ratio={vr:.4f} < {strict_thr:.2f} "
                    f"sample={b} img={img_paths[b]} mode={best_mode}"
                )

        stats = {
            "frame": int(frame_i),
            "valid_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
            "valid_ratio_min": float(np.min(valid_ratios)) if valid_ratios else 0.0,
            "pose_mode": str(best_modes[0]) if best_modes else "na",
            "feat_levels": [int(l) for l in feat_levels],
        }
        log_every = int(cfg.get("log_valid_every", 0))
        seen = int(getattr(self, "_gdino_point_seen", 0))
        if log_every > 0 and (seen % log_every == 0):
            self._log_mmengine(
                f"[GDINO][point_fusion][valid_ratio] frame={int(frame_i)} "
                f"mean={stats['valid_ratio_mean']:.4f} min={stats['valid_ratio_min']:.4f} mode={stats['pose_mode']}"
            )
        self._gdino_point_seen = seen + 1
        return out_feats, stats

    def _run_gdino_daca2d_for_frame(self, batch_inputs_dict, batch_data_samples, frame_i: int = 0, *, is_train: bool = False):
        """Object-level injection: run GDINO full forward and lift 2D queries to 3D anchors."""
        cfg = self._get_gdino_daca2d_cfg(is_train=is_train)
        if not (isinstance(cfg, dict) and bool(cfg.get("enable", False))):
            return None, None, None

        score_thr = float(cfg.get("score_thr", 0.25))
        max_queries = int(cfg.get("max_queries", 50))
        max_depth = float(cfg.get("max_depth", 10.0))
        min_support = int(cfg.get("query3d_center", {}).get("min_support_pts", 30))
        big_box_thr = float(cfg.get("support_stats", {}).get("big_box_area_px", 10000.0))

        pts_raw_in = batch_inputs_dict.get("points_raw", None)
        pts_aug_in = batch_inputs_dict.get("points", None)
        if pts_aug_in is None:
            return None, None, {"skipped": "no_points", "frame": int(frame_i)}
        B = len(batch_data_samples)
        device = pts_aug_in[0].device

        cams, img_paths = [], []
        for b in range(B):
            meta_b = getattr(batch_data_samples[b], "metainfo", None)
            if callable(meta_b):
                meta_b = meta_b()
            if not isinstance(meta_b, dict):
                meta_b = getattr(batch_data_samples[b], "img_metas", {}) or {}
            cam, img_path = self._unwrap_cam_and_img(meta_b, frame_i=int(frame_i))
            if cam is None or img_path is None:
                cams.append(None)
                img_paths.append(None)
                continue
            cams.append(cam)
            img_paths.append(img_path)

        hs_last = pred_boxes = pred_scores = None
        cache_dir = cfg.get("cache_dir", None)
        if cache_dir is None and isinstance(cfg.get("cache", None), dict):
            cache_dir = cfg.get("cache", {}).get("dir", None) or cfg.get("cache", {}).get("cache_dir", None)
        if cache_dir is None:
            cache_dir = os.environ.get("GDINO_CACHE_DIR", None)
        if cache_dir is not None and all(isinstance(p, str) for p in img_paths) and all(isinstance(c, dict) for c in cams):
            hw0 = cams[0].get("img_size_gdino", None)
            if torch.is_tensor(hw0) and hw0.numel() == 2:
                target_hw = (int(hw0.reshape(-1)[0].item()), int(hw0.reshape(-1)[1].item()))
                bb_cfg = None
                if isinstance(cfg.get("gdino", None), dict):
                    bb_cfg = cfg["gdino"].get("backbone", None)
                if not isinstance(bb_cfg, dict):
                    bb_cfg = getattr(self, "_gdino_backbone_cfg", None)
                cached = load_gdino_cache_batched(
                    cache_dir,
                    img_paths=[str(p) for p in img_paths],
                    target_hw=target_hw,
                    bb_cfg=bb_cfg if isinstance(bb_cfg, dict) else None,
                    mode="full",
                    device=device,
                    dtype=torch.float32,
                )
                if isinstance(cached, dict):
                    hs_last = cached.get("hs_last", None)
                    pred_boxes = cached.get("pred_boxes", None)
                    pred_scores = cached.get("pred_scores", None)

        if hs_last is None or pred_boxes is None or pred_scores is None:
            imgs = []
            for b in range(B):
                cam = cams[b]
                img_path = img_paths[b] if b < len(img_paths) else None
                if cam is None or img_path is None:
                    imgs.append(None)
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_size = cam.get("img_size_gdino", None)
                    if torch.is_tensor(img_size):
                        h1, w1 = int(img_size[0].item()), int(img_size[1].item())
                        if (img.height, img.width) != (h1, w1):
                            img = img.resize((w1, h1), resample=Image.BILINEAR)
                    imgs.append(img)
                except Exception:
                    imgs.append(None)

            if any(im is None for im in imgs):
                return None, None, {"skipped": "img_load_failed", "frame": int(frame_i)}

            img_t = torch.stack([self._to_tensor_img(im, device=device) for im in imgs], dim=0)
            gdino = self._get_gdino_backbone(cfg.get("gdino", None))
            out = gdino(img_t, backbone_only=False)
            hs_last = out.get("hs_last", None)
            pred_boxes = out.get("pred_boxes", None)
            pred_scores = out.get("pred_scores", None)
        if hs_last is None or pred_boxes is None or pred_scores is None:
            return None, None, {"skipped": "gdino_no_outputs", "frame": int(frame_i)}

        q2d_feats_list, q2d_pos_list = [], []
        nq_keep_list, nq_pos_list, valid_ratios = [], [], []
        for b in range(B):
            cam = cams[b]
            if cam is None:
                q2d_feats_list.append(torch.zeros((0, hs_last.shape[-1]), device=device))
                q2d_pos_list.append(torch.zeros((0, 3), device=device))
                nq_keep_list.append(0)
                nq_pos_list.append(0)
                valid_ratios.append(0.0)
                continue

            intr = cam.get("intrinsics", None)
            if isinstance(intr, (list, tuple)) and len(intr) == 1 and torch.is_tensor(intr[0]):
                intr_t = intr[0].to(device=device, dtype=torch.float32).reshape(-1)[:4]
            elif torch.is_tensor(intr):
                intr_t = intr.to(device=device, dtype=torch.float32).reshape(-1)[:4]
            else:
                intr_t = torch.as_tensor(intr, device=device, dtype=torch.float32).reshape(-1)[:4]
            fx, fy, cx, cy = [float(v) for v in intr_t.tolist()]
            pose = cam.get("pose", cam.get("extrinsics", None))
            if torch.is_tensor(pose):
                pose_t = pose.to(device=device, dtype=torch.float32).reshape(4, 4)
            else:
                pose_t = torch.as_tensor(pose, device=device, dtype=torch.float32).reshape(4, 4)
            img_hw = cam.get("img_size_gdino", None)
            if torch.is_tensor(img_hw):
                h_img, w_img = int(img_hw[0].item()), int(img_hw[1].item())
            else:
                h_img, w_img = int(img_t.shape[-2]), int(img_t.shape[-1])

            pts_aug = pts_aug_in[b]
            xyz_aug = pts_aug[:, :3]
            if isinstance(pts_raw_in, (list, tuple)) and len(pts_raw_in) == B:
                xyz_raw = pts_raw_in[b][:, :3]
            else:
                xyz_raw = xyz_aug

            def _ratio_for_mode(mode: str) -> float:
                if mode == "inv":
                    mat = torch.linalg.inv(pose_t)
                elif mode == "direct":
                    mat = pose_t
                else:
                    mat = torch.eye(4, device=device, dtype=pose_t.dtype)
                xyz1 = torch.cat([xyz_raw, torch.ones((xyz_raw.shape[0], 1), device=device, dtype=xyz_raw.dtype)], dim=1)
                xyz_cam = (xyz1 @ mat.T)[:, :3]
                x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
                valid_z = (z > float(MIN_DEPTH)) & (z < max_depth) & (torch.abs(z) > torch.finfo(z.dtype).eps)
                ratio_x = torch.zeros_like(x)
                ratio_y = torch.zeros_like(y)
                if valid_z.any():
                    ratio_x[valid_z] = x[valid_z] / z[valid_z]
                    ratio_y[valid_z] = y[valid_z] / z[valid_z]
                u_img = fx * ratio_x + cx
                v_img = fy * ratio_y + cy
                valid_u = (u_img >= 0) & (u_img < float(w_img))
                valid_v = (v_img >= 0) & (v_img < float(h_img))
                valid = valid_z & valid_u & valid_v
                return float(valid.float().mean().item())

            best_mode = "inv"
            best_ratio = _ratio_for_mode("inv")
            for m in ("direct", "identity"):
                r = _ratio_for_mode(m)
                if r > best_ratio:
                    best_ratio, best_mode = r, m
            valid_ratios.append(best_ratio)

            if best_mode == "inv":
                mat = torch.linalg.inv(pose_t)
            elif best_mode == "direct":
                mat = pose_t
            else:
                mat = torch.eye(4, device=device, dtype=pose_t.dtype)
            xyz1 = torch.cat([xyz_raw, torch.ones((xyz_raw.shape[0], 1), device=device, dtype=xyz_raw.dtype)], dim=1)
            xyz_cam = (xyz1 @ mat.T)[:, :3]
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            valid_z = (z > float(MIN_DEPTH)) & (z < max_depth) & (torch.abs(z) > torch.finfo(z.dtype).eps)
            ratio_x = torch.zeros_like(x)
            ratio_y = torch.zeros_like(y)
            if valid_z.any():
                ratio_x[valid_z] = x[valid_z] / z[valid_z]
                ratio_y[valid_z] = y[valid_z] / z[valid_z]
            u_img = fx * ratio_x + cx
            v_img = fy * ratio_y + cy
            uv_img = torch.stack([u_img, v_img], dim=-1)

            hs_b = hs_last[b]
            boxes_b = pred_boxes[b]
            scores_b = pred_scores[b]
            keep = scores_b >= score_thr
            if int(keep.sum().item()) > max_queries:
                topk = torch.topk(scores_b, k=max_queries, largest=True).indices
                keep = torch.zeros_like(scores_b, dtype=torch.bool)
                keep[topk] = True
            keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
            nq_keep = int(keep_idx.numel())
            nq_keep_list.append(nq_keep)
            if nq_keep == 0:
                q2d_feats_list.append(torch.zeros((0, hs_b.shape[-1]), device=device))
                q2d_pos_list.append(torch.zeros((0, 3), device=device))
                nq_pos_list.append(0)
                continue

            hs_keep = hs_b[keep_idx]
            boxes_keep = boxes_b[keep_idx]  # normalized cxcywh
            cxn, cyn, wn, hn = boxes_keep[:, 0], boxes_keep[:, 1], boxes_keep[:, 2], boxes_keep[:, 3]
            x0 = (cxn - 0.5 * wn) * float(w_img)
            x1 = (cxn + 0.5 * wn) * float(w_img)
            y0 = (cyn - 0.5 * hn) * float(h_img)
            y1 = (cyn + 0.5 * hn) * float(h_img)

            q_pos_acc, q_feat_acc = [], []
            support_counts = []
            box_areas = []
            for qi in range(nq_keep):
                in_box = (
                    valid_z
                    & (uv_img[:, 0] >= x0[qi])
                    & (uv_img[:, 0] <= x1[qi])
                    & (uv_img[:, 1] >= y0[qi])
                    & (uv_img[:, 1] <= y1[qi])
                )
                idx = torch.nonzero(in_box, as_tuple=False).squeeze(-1)
                cnt = int(idx.numel())
                support_counts.append(cnt)
                try:
                    box_areas.append(float(((x1[qi] - x0[qi]) * (y1[qi] - y0[qi])).abs().item()))
                except Exception:
                    box_areas.append(0.0)
                if cnt < min_support:
                    continue
                q_pos_acc.append(xyz_aug[idx].median(dim=0).values)
                q_feat_acc.append(hs_keep[qi])

            if len(q_pos_acc) == 0:
                q2d_feats_list.append(torch.zeros((0, hs_b.shape[-1]), device=device))
                q2d_pos_list.append(torch.zeros((0, 3), device=device))
                nq_pos_list.append(0)
                continue
            q2d_feats_list.append(torch.stack(q_feat_acc, dim=0))
            q2d_pos_list.append(torch.stack(q_pos_acc, dim=0))
            nq_pos_list.append(int(len(q_pos_acc)))

            # Aggregate support stats across batch for diagnostics.
            try:
                if "support_counts_all" not in locals():
                    support_counts_all = []
                    box_areas_all = []
                support_counts_all.extend([int(x) for x in support_counts])
                box_areas_all.extend([float(x) for x in box_areas])
            except Exception:
                pass

        stats = {
            "frame": int(frame_i),
            "valid_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
            "valid_ratio_min": float(np.min(valid_ratios)) if valid_ratios else 0.0,
            "qpos_rate_mean": float(np.mean([p / max(k, 1) for p, k in zip(nq_pos_list, nq_keep_list)])) if nq_keep_list else 0.0,
            "qpos_rate_min": float(np.min([p / max(k, 1) for p, k in zip(nq_pos_list, nq_keep_list)])) if nq_keep_list else 0.0,
            "nq_keep_mean": float(np.mean(nq_keep_list)) if nq_keep_list else 0.0,
            "nq_pos_mean": float(np.mean(nq_pos_list)) if nq_pos_list else 0.0,
        }

        # ---- support stats (in-box support points per 2D box) ----
        try:
            sc = np.asarray(locals().get("support_counts_all", []), dtype=np.float32)
            ba = np.asarray(locals().get("box_areas_all", []), dtype=np.float32)
            if sc.size > 0:
                stats.update(
                    {
                        "min_support": int(min_support),
                        "support_p50": float(np.percentile(sc, 50)),
                        "support_p90": float(np.percentile(sc, 90)),
                        "support_lt_min_rate": float(np.mean(sc < float(min_support))),
                        "box_area_p50": float(np.percentile(ba, 50)) if ba.size > 0 else 0.0,
                        "box_area_p90": float(np.percentile(ba, 90)) if ba.size > 0 else 0.0,
                        "big_box_thr": float(big_box_thr),
                    }
                )
                if ba.size == sc.size and ba.size > 0:
                    big = ba >= float(big_box_thr)
                    stats["big_box_rate"] = float(np.mean(big)) if big.size > 0 else 0.0
                    stats["big_box_low_support_rate"] = float(
                        np.mean((sc < float(min_support)) & big)
                    ) if np.any(big) else 0.0
        except Exception:
            pass
        log_every = int(cfg.get("log_valid_every", 50))
        seen = int(getattr(self, "_gdino_daca2d_seen", 0))
        if log_every > 0 and (seen % log_every == 0):
            self._log_mmengine(
                f"[GDINO][daca2d][valid_ratio] frame={int(frame_i)} "
                f"mean={stats['valid_ratio_mean']:.4f} min={stats['valid_ratio_min']:.4f} "
                f"qpos_rate_mean={stats['qpos_rate_mean']:.4f} qpos_rate_min={stats['qpos_rate_min']:.4f} "
                f"nq_keep_mean={stats['nq_keep_mean']:.1f} nq_pos_mean={stats['nq_pos_mean']:.1f} "
                f"support_p50={stats.get('support_p50', 0.0):.1f} support_p90={stats.get('support_p90', 0.0):.1f} "
                f"lt_min={stats.get('support_lt_min_rate', 0.0):.3f} "
                f"big_low={stats.get('big_box_low_support_rate', 0.0):.3f}"
            )
        self._gdino_daca2d_seen = seen + 1
        return q2d_feats_list, q2d_pos_list, stats

    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        """Extract features from sparse tensor.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.

        Returns:
            Tuple:
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_channels).
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_classes + 1).
        """
        # construct tensor field
        coordinates, features = [], []
        coordinates_wo_elastic = []

        # Optional: point-wise GDINO features.
        # Two modes:
        #   - early-fusion: concat/add to RGB features before SparseTensor
        #   - sparse-fpn: build_sparse_fpn(coords, feats) and inject into UNet decoder (no in_channels change)
        gdino_point_feats = None
        gdino_point_stats = None
        try:
            gdino_point_feats, gdino_point_stats = self._run_gdino_point_fusion_for_frame(
                batch_inputs_dict, batch_data_samples, frame_i=0
            )
        except Exception as e:
            gdino_point_feats, gdino_point_stats = None, {"frame": 0, "skipped": "exception", "error": repr(e)}
            if isinstance(self.gdino_point_fusion_cfg, dict) and bool(self.gdino_point_fusion_cfg.get("log_fail", False)):
                print(f"[GDINO][point_fusion][error] frame=0 err={repr(e)}")
        self._last_gdino_point_fusion_stats = gdino_point_stats

        enable_pf = bool(self.gdino_point_fusion_cfg.get("enable", False)) if isinstance(self.gdino_point_fusion_cfg, dict) else False
        fuse_mode = str(self.gdino_point_fusion_cfg.get("fuse_mode", "concat")).lower() if isinstance(self.gdino_point_fusion_cfg, dict) else "concat"
        pf_mode = str(self.gdino_point_fusion_cfg.get("mode", "early")).lower() if isinstance(self.gdino_point_fusion_cfg, dict) else "early"
        if fuse_mode in ("fpn", "sparse_fpn"):
            pf_mode = "fpn"

        gdino_sparse_fpn = None
        if enable_pf:
            ok = isinstance(gdino_point_feats, (list, tuple)) and len(gdino_point_feats) == len(batch_inputs_dict.get("points", []))
            if not ok:
                raise RuntimeError(
                    "[GDINO][point_fusion] enabled but no per-point features were produced. "
                    f"stats={gdino_point_stats}"
                )
            if pf_mode == "fpn":
                # Build sparse pyramid from point-wise feats and the same coord source as backbone.
                # This keeps UNet in_channels unchanged (A3/A4) and matches ESAM DINO-FPN semantics.
                try:
                    coords_list, feats_list = [], []
                    for b in range(len(batch_inputs_dict["points"])):
                        if "elastic_coords" in batch_inputs_dict and batch_inputs_dict["elastic_coords"] is not None:
                            e = batch_inputs_dict["elastic_coords"][b]
                            coords = torch.floor(e).to(torch.int32)
                        else:
                            xyz = batch_inputs_dict["points"][b][:, :3]
                            coords = torch.floor(xyz / float(self.voxel_size)).to(torch.int32)
                        batch_col = torch.full((coords.shape[0], 1), b, dtype=torch.int32, device=coords.device)
                        coords_batched = torch.cat([batch_col, coords], dim=1)
                        coords_list.append(coords_batched)
                        feats_list.append(gdino_point_feats[b].to(device=coords.device))
                    coords_batch = torch.cat(coords_list, dim=0)
                    feats_batch = torch.cat(feats_list, dim=0)
                    gdino_sparse_fpn = build_sparse_fpn(coords_batch, feats_batch)
                except Exception as e:
                    raise RuntimeError(f"[GDINO][point_fusion][fpn] build_sparse_fpn failed: {repr(e)}")

        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][:, :3])
            # Keep a wo-elastic coordinate copy for DACA-2D distance gating.
            coordinates_wo_elastic.append(batch_inputs_dict['points'][i][:, :3])
            feat_i = batch_inputs_dict['points'][i][:, 3:]
            if enable_pf and pf_mode != "fpn":
                try:
                    if fuse_mode in ("concat", "cat"):
                        feat_i = torch.cat([feat_i, gdino_point_feats[i]], dim=1)
                    elif fuse_mode in ("add", "add_rgb", "sum"):
                        if feat_i.shape[1] != gdino_point_feats[i].shape[1]:
                            raise RuntimeError(
                                f"add-mode requires same dim, got rgb_dim={feat_i.shape[1]} gdino_dim={gdino_point_feats[i].shape[1]}"
                            )
                        feat_i = feat_i + gdino_point_feats[i]
                    else:
                        raise RuntimeError(f"unknown fuse_mode={fuse_mode}")
                except Exception as e:
                    raise RuntimeError(
                        "[GDINO][point_fusion] feature concat failed: "
                        f"rgb_shape={tuple(feat_i.shape)} gdino_shape={tuple(getattr(gdino_point_feats[i], 'shape', ()))}, "
                        f"err={repr(e)}"
                    )
            features.append(feat_i)
        all_xyz = coordinates
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        if gdino_sparse_fpn is not None:
            x = self.backbone(field.sparse(), dino_feats=gdino_sparse_fpn)  # type: ignore[call-arg]
        else:
            x = self.backbone(field.sparse())  # [N_segment, 96]
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)] # [batch_size * 20000, 96]
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            # Robustness: compact possibly gappy superpoint ids to [0..N-1]
            # so scatter ops don't allocate by a huge max().
            sp_pts_mask = sp_pts_mask.to(dtype=torch.long)
            _, sp_pts_mask = torch.unique(sp_pts_mask, sorted=True, return_inverse=True)
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(int(sp_pts_mask.max().item()) + 1 if sp_pts_mask.numel() else 0)
        sp_idx = torch.cat(sp_pts_masks)

        # Precompute SP positions in elastic space (used by bbox losses / box-CA3D).
        # This must be consistent with the backbone/decoder space: elastic_coords*voxel_size if present, else points[:,:3].
        try:
            xyz_elastic = torch.cat(all_xyz, dim=0)
            sp_xyz_elastic = scatter_mean(xyz_elastic, sp_idx, dim=0)
            sp_pos_elastic_list = []
            start = 0
            for n_sp in n_super_points:
                end = start + int(n_sp)
                sp_pos_elastic_list.append(sp_xyz_elastic[start:end])
                start = end
            self._last_sp_pos_elastic_list = sp_pos_elastic_list
        except Exception:
            self._last_sp_pos_elastic_list = None

        x, all_xyz_w = self.pool(x, sp_idx, all_xyz)

        # Precompute SP positions in wo-elastic space for DACA-2D distance gating.
        try:
            xyz_wo = torch.cat(coordinates_wo_elastic, dim=0)
            sp_xyz_wo = scatter_mean(xyz_wo, sp_idx, dim=0)
            sp_pos_list = []
            start = 0
            for n_sp in n_super_points:
                end = start + int(n_sp)
                sp_pos_list.append(sp_xyz_wo[start:end])
                start = end
            self._last_sp_pos_wo_elastic_list = sp_pos_list
        except Exception:
            self._last_sp_pos_wo_elastic_list = None

        # apply cls_layer
        features = []
        for i in range(len(n_super_points)): # batch_size
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
        return features, point_features, all_xyz_w

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        ## Backbone
        x, point_features, all_xyz_w = self.extract_feat(batch_inputs_dict, batch_data_samples) # batch_size * [N_segment, 96] batch_size * [200000, 99] [batch_size * 200000, 1] 
        ## GT-prepare
        gt_instances = [s.gt_instances_3d for s in batch_data_samples]
        gt_point_instances = []
        for i in range(len(gt_instances)): # batch_size
            ins = batch_data_samples[i].gt_pts_seg.pts_instance_mask # [20000]
            if torch.sum(ins == -1) != 0:
                ins[ins == -1] = torch.max(ins) + 1
                ins = F.one_hot(ins)[:, :-1]
            else:
                ins = F.one_hot(ins)
            ins = ins.bool().T
            gt_point = InstanceData()
            gt_point.p_masks = ins
            gt_point_instances.append(gt_point)

        # Optional: compute GT axis-aligned bboxes/centers/sizes + scene_range on-the-fly
        # (SV infos often omit bboxes_3d). This is only used when enabled explicitly in config.
        self._last_scene_range = None
        try:
            if isinstance(self.train_cfg, dict) and bool(self.train_cfg.get('compute_gt_bboxes_3d', False)):
                scene_range = []
                for b in range(len(gt_instances)):
                    masks = gt_point_instances[b].p_masks  # (n_inst, n_pts)
                    device = x[b].device
                    # Use the same xyz space as decoder/box losses (elastic coords if present).
                    if 'elastic_coords' in batch_inputs_dict and batch_inputs_dict['elastic_coords'] is not None:
                        xyz = batch_inputs_dict['elastic_coords'][b].to(device) * float(self.voxel_size)
                    else:
                        xyz = batch_inputs_dict['points'][b][:, :3].to(device)

                    # Scene range for positional encoding normalization.
                    if xyz.numel() == 0:
                        smin = torch.zeros((3,), device=device, dtype=torch.float32)
                        smax = torch.ones((3,), device=device, dtype=torch.float32)
                    else:
                        smin = xyz.min(dim=0).values
                        smax = xyz.max(dim=0).values
                    scene_range.append((smin, smax))

                    n_inst = int(masks.shape[0]) if masks.ndim == 2 else 0
                    if n_inst == 0 or xyz.numel() == 0:
                        bboxes = torch.zeros((0, 7), device=device, dtype=torch.float32)
                        centers = torch.zeros((0, 3), device=device, dtype=torch.float32)
                        sizes = torch.zeros((0, 3), device=device, dtype=torch.float32)
                    else:
                        b_list, c_list, s_list = [], [], []
                        for mi in range(n_inst):
                            idx = torch.nonzero(masks[mi], as_tuple=False).squeeze(-1)
                            if idx.numel() == 0:
                                c = torch.zeros((3,), device=device, dtype=torch.float32)
                                sz = torch.zeros((3,), device=device, dtype=torch.float32)
                            else:
                                pts = xyz[idx]
                                pmin = pts.min(dim=0).values
                                pmax = pts.max(dim=0).values
                                c = (pmin + pmax) * 0.5
                                sz = (pmax - pmin).clamp(min=0)
                            b_list.append(torch.cat([c, sz, torch.zeros((1,), device=device, dtype=torch.float32)], dim=0))
                            c_list.append(c)
                            s_list.append(sz)
                        bboxes = torch.stack(b_list, dim=0)  # (n_inst, 7)
                        centers = torch.stack(c_list, dim=0)
                        sizes = torch.stack(s_list, dim=0)

                    # Pad to match sp_masks rows (instances + semantic rows), keep naming aligned to SegDINO3D.
                    try:
                        n_rows = int(gt_instances[b].sp_masks.shape[0])
                    except Exception:
                        n_rows = int(bboxes.shape[0])
                    if n_rows > bboxes.shape[0]:
                        pad7 = torch.zeros((n_rows - bboxes.shape[0], 7), device=device, dtype=bboxes.dtype)
                        bboxes = torch.cat([bboxes, pad7], dim=0)
                        pad3 = torch.zeros((n_rows - centers.shape[0], 3), device=device, dtype=centers.dtype)
                        centers = torch.cat([centers, pad3], dim=0)
                        sizes = torch.cat([sizes, pad3], dim=0)

                    # Only overwrite if missing to avoid clobbering dataset-provided boxes.
                    if not (hasattr(gt_instances[b], 'bboxes_3d') and gt_instances[b].bboxes_3d is not None):
                        gt_instances[b].bboxes_3d = bboxes
                    gt_instances[b].instance_centers = centers
                    gt_instances[b].instance_sizes = sizes

                self._last_scene_range = scene_range
        except Exception:
            self._last_scene_range = None

        # Select queries and keep query centers aligned (needed by bbox losses / Center&Size costs).
        sp_pos_elastic_list = getattr(self, "_last_sp_pos_elastic_list", None)
        if sp_pos_elastic_list is not None:
            queries, gt_instances, sp_pos_elastic_list = self._select_queries(x, gt_instances, sp_pos_elastic_list)
        else:
            queries, gt_instances = self._select_queries(x, gt_instances) # 随机选出 0.5 ~ 1数量的query
        ## Decoder
        super_points = ([bds.gt_pts_seg.sp_pts_mask for bds in batch_data_samples], all_xyz_w) # 每个点的segment ID以及归一化权重

        # Optional: GDINO DACA-2D object-level injection (SP-domain only in decoder).
        query2d_feats = query2d_pos = None
        gdino_daca2d_cfg = None
        try:
            gdino_daca2d_cfg = self._get_gdino_daca2d_cfg(is_train=True)
            if isinstance(gdino_daca2d_cfg, dict) and bool(gdino_daca2d_cfg.get('enable', False)):
                q2d_f, q2d_p, _stats = self._run_gdino_daca2d_for_frame(
                    batch_inputs_dict, batch_data_samples, frame_i=0, is_train=True
                )
                # Only inject when mode == 'fuse'; diag_only only records stats.
                if str(gdino_daca2d_cfg.get('mode', 'fuse')).lower() == 'fuse':
                    query2d_feats, query2d_pos = q2d_f, q2d_p
        except Exception as e:
            # Keep training robust: fail closed (no injection).
            if isinstance(gdino_daca2d_cfg, dict) and bool(gdino_daca2d_cfg.get('strict', False)):
                raise
            if isinstance(gdino_daca2d_cfg, dict) and bool(gdino_daca2d_cfg.get('log_fail', False)):
                print(f"[GDINO][daca2d][error] frame=0 err={repr(e)}")

        x = self.decoder(
            x, point_features, queries, super_points, use_one2many=self.use_one2many,
            query2d_feats=query2d_feats, query2d_pos=query2d_pos, gdino_daca2d_cfg=gdino_daca2d_cfg,
            sp_pos_list_override=getattr(self, '_last_sp_pos_wo_elastic_list', None),
            # Box-modulated CA-3D (optional): needs elastic SP positions + scene range in the same space.
            sp_pos_list_elastic=sp_pos_elastic_list,
            scene_range=getattr(self, "_last_scene_range", None),
            query3d_pos=sp_pos_elastic_list,
        ) # [N_segment, 96] [20000, 99] [(0.5 ~ 1) * N_segment, 96] ([20000, 1])
        loss = self.criterion(x, gt_instances, gt_point_instances, sp_pos_elastic_list, self.decoder.mask_pred_mode)
        if self.use_one2many:
            loss_one2many = self.criterion_one2many(x['one2many_outputs'], gt_instances, gt_point_instances, None, self.decoder.mask_pred_mode, use_one2many=self.use_one2many)
            for key, value in loss_one2many.items():
                loss_one2many[key] = value * self.one2many_loss_weight
            loss.update(loss_one2many)
        ## Loss
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        assert len(batch_data_samples) == 1
        ## Backbone
        x, point_features, all_xyz_w = self.extract_feat(batch_inputs_dict, batch_data_samples)
        ## Decoder
        super_points = ([bds.gt_pts_seg.sp_pts_mask for bds in batch_data_samples], all_xyz_w)

        query2d_feats = query2d_pos = None
        gdino_daca2d_cfg = None
        try:
            gdino_daca2d_cfg = self._get_gdino_daca2d_cfg(is_train=False)
            if isinstance(gdino_daca2d_cfg, dict) and bool(gdino_daca2d_cfg.get('enable', False)):
                q2d_f, q2d_p, _stats = self._run_gdino_daca2d_for_frame(
                    batch_inputs_dict, batch_data_samples, frame_i=0, is_train=False
                )
                if str(gdino_daca2d_cfg.get('mode', 'fuse')).lower() == 'fuse':
                    query2d_feats, query2d_pos = q2d_f, q2d_p
        except Exception as e:
            if isinstance(gdino_daca2d_cfg, dict) and bool(gdino_daca2d_cfg.get('strict', False)):
                raise
            if isinstance(gdino_daca2d_cfg, dict) and bool(gdino_daca2d_cfg.get('log_fail', False)):
                print(f"[GDINO][daca2d][error] frame=0 err={repr(e)}")

        x = self.decoder(
            x, point_features, x, super_points,
            query2d_feats=query2d_feats, query2d_pos=query2d_pos, gdino_daca2d_cfg=gdino_daca2d_cfg,
            sp_pos_list_override=getattr(self, '_last_sp_pos_wo_elastic_list', None),
        )
        ## Post-processing
        pred_pts_seg = self.predict_by_feat(
            x, batch_data_samples[0].gt_pts_seg.sp_pts_mask)
        # Optional: lightweight per-scene monitor for SV (mirrors Online class structure).
        try:
            mon_cfg = (self.test_cfg.get('online_monitor', None) or {}) if hasattr(self, 'test_cfg') else {}
            if bool(mon_cfg.get('enable', False)):
                meta = getattr(batch_data_samples[0], 'img_metas', None)
                if not isinstance(meta, dict):
                    try:
                        meta = batch_data_samples[0].metainfo
                    except Exception:
                        meta = {}
                scene_id = (
                    meta.get('scene_id', None)
                    or meta.get('scan_id', None)
                    or meta.get('sample_idx', None)
                    or meta.get('lidar_idx', None)
                    or meta.get('ann_file', None)
                    or meta.get('pts_filename', None)
                    or 'unknown'
                )
                fr = {'frame': 0}
                if isinstance(getattr(self, '_last_gdino_daca2d_stats', None), dict):
                    fr['gdino_daca2d'] = self._last_gdino_daca2d_stats
                if isinstance(getattr(self, '_last_gdino_point_fusion_stats', None), dict):
                    fr['gdino'] = self._last_gdino_point_fusion_stats
                try:
                    st = getattr(self.decoder, '_last_daca2d_apply_stats', None)
                    if isinstance(st, dict) and isinstance(st.get('agg', None), dict):
                        agg = st['agg']
                        nq2d = float(agg.get('nq2d_mean', 0.0) or 0.0)
                        allowed_mean = float(agg.get('allowed_q2d_mean', 0.0) or 0.0)
                        fr['daca2d_apply'] = {
                            'allowed_zero_rate': float(agg.get('allowed_q2d_zero_rate', 0.0) or 0.0),
                            'allowed_q2d_mean': allowed_mean,
                            'allowed_q2d_p50': float(agg.get('allowed_q2d_p50', 0.0) or 0.0),
                            'allowed_q2d_p90': float(agg.get('allowed_q2d_p90', 0.0) or 0.0),
                            'nq2d': nq2d,
                            'nq3d': float(agg.get('nq3d_mean', 0.0) or 0.0),
                            'density': float(allowed_mean / (nq2d + 1e-6)) if nq2d > 0 else 0.0,
                            'delta_rel_mean': float(agg.get('delta_rel_mean', 0.0) or 0.0),
                            'q2d_any_sp_rate': float(agg.get('q2d_any_sp_rate', 0.0) or 0.0),
                        }
                except Exception:
                    pass
                pred_pts_seg[0].online_monitor = {
                    'scene_id': str(scene_id),
                    'num_frames': 1,
                    'frames': [fr],
                }
        except Exception:
            pass
        batch_data_samples[0].pred_pts_seg = pred_pts_seg[0]
        return batch_data_samples
    
    def predict_by_feat_instance(
            self,
            out,
            superpoints,
            score_threshold,
            return_queries: bool = False):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
                Tensor: instance queries of shape (n_preds, d), if
                    `return_queries=True`.
        """
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        queries = out.get('queries', None)
        queries = None if queries is None else queries[0]
        assert self.num_classes == 1 or self.num_classes == cls_preds.shape[1] - 1

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        if self.num_classes == 1:
            scores = scores.sum(-1, keepdim=True)
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        topk_num = min(self.test_cfg.topk_insts, scores.shape[0] * scores.shape[1])
        scores, topk_idx = scores.flatten(0, 1).topk(topk_num, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        if return_queries:
            if queries is None:
                raise RuntimeError(
                    '[predict_by_feat_instance] return_queries=True but out[\"queries\"] is missing.')
            query_pred = queries[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, keep_inds = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)
            if return_queries:
                query_pred = query_pred[keep_inds]

        mask_pred_sigmoid = mask_pred_sigmoid[:, ...]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]
        if return_queries:
            query_pred = query_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]
        if return_queries:
            query_pred = query_pred[npoint_mask]

        if return_queries:
            return mask_pred, labels, scores, query_pred
        return mask_pred, labels, scores

@MODELS.register_module()
class ScanNet200MixFormer3D_FF(ScanNet200MixFormer3D):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 img_backbone=None,
                 backbone=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.img_backbone = MODELS.build(img_backbone)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.pool = MODELS.build(pool)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(960, 32, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True))
    
    def init_weights(self):
        if hasattr(self, 'img_backbone'):
            self.img_backbone.init_weights()
    
    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        """Extract features from sparse tensor.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.

                
        Returns:
            Tuple:
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_channels).
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_classes + 1).
        """
        # extract image features
        with torch.no_grad():
            img_features = self.img_backbone(batch_inputs_dict['img_path']) # batch_size * [C, H, W][960, 60, 80]
        img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]
        
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict: # False
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][:, :3])
            features.append(batch_inputs_dict['points'][i][:, 3:])
        all_xyz = coordinates
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse(),
                          partial(self._f, img_features=img_features, img_metas=img_metas, img_shape=img_metas[0]['img_shape']))
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)]
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            sp_pts_mask = sp_pts_mask.to(dtype=torch.long)
            _, sp_pts_mask = torch.unique(sp_pts_mask, sorted=True, return_inverse=True)
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(int(sp_pts_mask.max().item()) + 1 if sp_pts_mask.numel() else 0)
        sp_idx = torch.cat(sp_pts_masks)
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz)

        # apply cls_layer
        features = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
        return features, point_features, all_xyz_w

    def _f(self, x, img_features, img_metas, img_shape):
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):
            coord_type = 'DEPTH'
            img_scale_factor = (
                point.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            #img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_flip = False
            img_crop_offset = (
                point.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_shape[-2:],
                img_shape=img_shape[-2:],
                aligned=True,
                padding_mode='zeros',
                align_corners=True))
 
        projected_features = torch.cat(projected_features, dim=0)
        projected_features = ME.SparseTensor(
            projected_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        
        projected_features = self.conv(projected_features)
        return projected_features + x

@MODELS.register_module()
class ScanNet200MixFormer3D_Online(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 map_to_rec_pcd=True,
                 backbone=None,
                 memory=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 merge_head=None,
                 merge_criterion=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 gdino_backbone: Optional[dict] = None,
                 gdino_point_fusion: Optional[dict] = None,

                 use_query_memory=False,
                 use_self_attn=False,
                 use_noise=False,
                 noise_p=0.05,
                 noise_k=10,
                 use_temporal_loss=False,
                 use_decouple=False,
                 use_mot=False,
                 mot_type='motr',
                 train_asso_only=False,
                 matcher=None,
                 use_aug=False,
                 asso_loss_weight=0.5,
                 use_refine=False,
                 asso_config=None,
                 use_one2many=False,
                 criterion_one2many=None,
                 use_3d_refine=False,
                 reweight_dict=None,
                 use_relative_asso=False,
                 merge_sp_masks = False,
                 replace_bn_with_ln=False,
                 debug_mode=False
                 ):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if memory is not None:
            self.memory = MODELS.build(memory)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.pool = MODELS.build(pool)
        self.decoder = MODELS.build(decoder)
        if merge_head is not None:
            self.merge_head = MODELS.build(merge_head)
        if merge_criterion is not None:
            self.merge_criterion = MODELS.build(merge_criterion)
        self.criterion = MODELS.build(criterion)
        self.decoder_online = decoder['temporal_attn']
        self.use_bbox = decoder['bbox_flag']
        self.sem_len = decoder['num_semantic_classes'] + 1 # 201
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.map_to_rec_pcd = map_to_rec_pcd
        # Debug/diagnostic flags (keep off by default unless config enables).
        self._log_missing_bboxes = bool(
            (train_cfg or {}).get("log_missing_bboxes", False)
            or (test_cfg or {}).get("log_missing_bboxes", False)
        )

        # Optional: GDINO backbone config (shared by gdino_diag/gdino_daca2d/gdino_point_fusion).
        self._gdino_backbone_cfg = gdino_backbone or {}
        # Optional: SegDINO3D-style point-wise 2D feature fusion into 3D backbone input.
        # This changes the backbone input channels, so keep fully gated in config.
        self.gdino_point_fusion_cfg = gdino_point_fusion or {}
        self._gdino_point_proj = None
        try:
            if isinstance(self.gdino_point_fusion_cfg, dict) and bool(self.gdino_point_fusion_cfg.get('enable', False)):
                in_dim = int(self.gdino_point_fusion_cfg.get('in_dim', 256))
                out_dim = int(self.gdino_point_fusion_cfg.get('out_dim', 32))
                proj_type = str(self.gdino_point_fusion_cfg.get('proj_type', 'linear')).lower()
                if proj_type == 'identity':
                    self._gdino_point_proj = nn.Identity()
                else:
                    self._gdino_point_proj = nn.Linear(in_dim, out_dim, bias=False)
        except Exception:
            self._gdino_point_proj = None
        # self.use_query_memory = decoder['use_query_memory']
        self.use_query_memory = use_query_memory
        if self.use_query_memory:
            self.muti_scale_query = MultiScaleQuery()
            self.query_memory = None
            self.pos_memory = None
            self.query_memory_relu = nn.ReLU()
            self.muti_scale_query.init_weights()
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.muti_scale_self_attn = MultiScaleQuery()
            self.self_attn_relu = nn.ReLU()
            self.muti_scale_self_attn.init_weights()
        self.use_noise = use_noise
        if self.use_noise:
            self.noise_p = noise_p
            self.noise_k = noise_k
        self.use_temporal_loss = use_temporal_loss
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_mask_memory = None
            self.before_sp_xyz = None
            self.before_query_ids = None
        self.use_decouple = use_decouple
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_query_boxes = None
        self.use_mot = use_mot
        if self.use_mot:
            self.mot_type = mot_type
            if mot_type == 'dq_track':
                self.use_relative_asso = use_relative_asso
                if self.use_relative_asso:
                    self.embed_trans2 = nn.Linear(256, 1)
                    self.heatmap_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
                self.use_refine = use_refine
                # self.asso_loss_weight = asso_loss_weight
                self.iou_calculator = AxisAlignedBboxOverlaps3D()
                self.matcher = TASK_UTILS.build(matcher)
                self.tracklet_trans = DQ_FFN(d_model=256, d_ffn=256, dropout=0)
                self.detector_trans = DQ_FFN(d_model=256, d_ffn=256, dropout=0)
                self.box_trans = nn.Sequential(
                    nn.Linear(6, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
                # query_trans = {'with_att': True, 'with_pos': True, 'min_channels': 256, 'drop_rate': 0.0}
                query_trans = asso_config['query_trans'] if asso_config is not None else {'with_att': True, 'with_pos': True, 'min_channels': 256, 'drop_rate': 0.0}
                self.query_inter = QueryInteractionX(in_channels=256, mid_channels=256, **query_trans)
                # self.update_type = asso_config['update_type'] if asso_config is not None else 'ema'
                self.asso_config = EasyDict(asso_config)
                
                self.rel_dist_embed = nn.Sequential(
                    nn.Linear(1, 256),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
                self.embed_trans = nn.Linear(256, 1)
                loss_asso = {'use_sigmoid': False, 'loss_weight': 1.0}
                from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
                self.loss_asso = CrossEntropyLoss(**loss_asso)
                self.ema_decay_rate = 0.5
                # self.train_asso_only = train_asso_only
            else:
                raise NotImplementedError(f"mot_type {mot_type} is not supported")

        self.use_one2many = use_one2many
        if self.use_one2many:
            self.one2many_loss_weight = 0.5
            self.criterion_one2many = MODELS.build(criterion_one2many[0])
        self.reweight_dict = reweight_dict
        self.merge_sp_masks = merge_sp_masks
        if self.merge_sp_masks:
            self.acc_dict = {}
            self.merge_box_trans = nn.Sequential(
                nn.Linear(6, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                DQ_FFN(d_model=256, d_ffn=256, dropout=0))
            self.merge_dist_embed = nn.Sequential(
                    nn.Linear(1, 256),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
            self.merge_embed_trans = nn.Linear(256, 1)
            self.merge_heatmap_loss = nn.BCEWithLogitsLoss(reduction='mean')
            self.merge_iou_calculator = AxisAlignedBboxOverlaps3D()
            self.fuse_linear = nn.Sequential(
                nn.Linear(512, 256, bias=False),
                nn.ReLU(inplace=True),
                nn.LayerNorm(256)
            )
            self.merge_fusion = MergeFusion(256)
            query_trans = {'with_att': True, 'with_pos': False, 'min_channels': 256, 'drop_rate': 0.0}
            self.merge_query_inter = QueryInteractionX(in_channels=256, mid_channels=256, **query_trans)
        self._prev_param_snapshot = None
        if replace_bn_with_ln:
            replace_bn(self)
        self.debug_mode = debug_mode
        self.init_weights()
    
    def init_weights(self):
        if hasattr(self, 'memory'):
            self.memory.init_weights()
            
    def reset_query_memory(self):
        """Reset the detector.
        """
        if self.use_query_memory:
            self.query_memory = None
            self.pos_memory = None

    def _get_gdino_backbone(self, cfg=None):
        """Lazily build GroundingDINO backbone for diagnostics / DACA-2D.

        This is a side-channel module and must NOT affect model outputs.
        """
        bb = getattr(self, "_gdino_backbone", None)
        if bb is not None:
            return bb
        if cfg is None:
            cfg = {}
            try:
                cfg = (self.test_cfg or {}).get("gdino_diag", {}) or {}
            except Exception:
                cfg = {}
        bb_cfg = cfg.get("backbone", None) if isinstance(cfg, dict) else None
        if not isinstance(bb_cfg, dict) and isinstance(getattr(self, "_gdino_backbone_cfg", None), dict):
            # Allow configuring the backbone at model init: `gdino_backbone=dict(type=..., ...)`.
            bb_cfg = getattr(self, "_gdino_backbone_cfg")
        if isinstance(bb_cfg, dict) and len(bb_cfg) > 0:
            bb = MODELS.build(bb_cfg)
        else:
            from .gdino_backbone import GroundingDINOBackbone
            bb = GroundingDINOBackbone()
        self._gdino_backbone = bb
        return bb

    @staticmethod
    def _safe_get_img_metas(batch_data_samples):
        if not batch_data_samples:
            return {}
        meta = getattr(batch_data_samples[0], "img_metas", None)
        if isinstance(meta, dict):
            return meta
        try:
            meta = batch_data_samples[0].metainfo
            if isinstance(meta, dict):
                return meta
        except Exception:
            pass
        return {}

    def _run_gdino_diag_for_frame(self, batch_inputs_dict, batch_data_samples, frame_i: int):
        """Compute GDINO 2D-3D projection sanity stats (valid_ratio).

        Returns a small JSON-serializable dict, or None if disabled.
        """
        cfg = {}
        try:
            cfg = (self.test_cfg or {}).get("gdino_diag", {}) or {}
        except Exception:
            cfg = {}
        if not (isinstance(cfg, dict) and bool(cfg.get("enable", False))):
            return None

    @staticmethod
    def _quantile_stats(x: torch.Tensor, prefix: str = "", qs=(0.1, 0.5, 0.9)):
        """Return quantile stats for a 1D tensor as JSON-serializable dict."""
        out = {}
        if x is None:
            return out
        if not torch.is_tensor(x):
            try:
                x = torch.as_tensor(x)
            except Exception:
                return out
        x = x.flatten()
        out[f"{prefix}n"] = int(x.numel())
        if x.numel() == 0:
            for q in qs:
                out[f"{prefix}p{int(q*100):02d}"] = 0.0
            out[f"{prefix}mean"] = 0.0
            return out
        x = x.float()
        try:
            qv = torch.quantile(x, torch.tensor(list(qs), device=x.device, dtype=x.dtype))
            for idx, q in enumerate(qs):
                out[f"{prefix}p{int(q*100):02d}"] = float(qv[idx].item())
        except Exception:
            # Fallback: just provide mean.
            for q in qs:
                out[f"{prefix}p{int(q*100):02d}"] = 0.0
        out[f"{prefix}mean"] = float(x.mean().item())
        return out

    def _run_gt_emb_diag_for_frame(
        self,
        gt_inst: torch.Tensor,
        pred_masks: torch.Tensor,
        pred_queries: torch.Tensor,
        frame_i: int,
        state: dict,
        cfg: dict,
    ):
        """GT-aligned embedding stability diagnostic (oracle association).

        Computes:
        - Positive pairs: cosine(q_g(t), q_g(t-1)) for GT instances g matched in consecutive frames.
        - Negative pairs: cosine between different GT instance embeddings in the same frame.

        This is monitoring-only and must not affect prediction results.
        """
        if not (isinstance(cfg, dict) and bool(cfg.get("enable", False))):
            return None

        frame_stride = int(cfg.get("frame_stride", 1))
        if frame_stride > 1 and (int(frame_i) % frame_stride) != 0:
            return {"frame": int(frame_i), "skipped": "stride"}

        min_iou = float(cfg.get("min_iou", cfg.get("iou_lo_thr", 0.1)))
        iou_thr = float(cfg.get("iou_thr", 0.5))
        gt_vis_npoint = int(cfg.get("gt_vis_npoint", 100))
        neg_pairs = int(cfg.get("neg_pairs", 2048))
        save_raw = bool(cfg.get("save_raw", False))
        max_raw = int(cfg.get("max_raw", 4096))

        if gt_inst is None or (torch.is_tensor(gt_inst) and gt_inst.numel() == 0):
            return {"frame": int(frame_i), "skipped": "no_gt"}
        if pred_masks is None or pred_queries is None:
            return {"frame": int(frame_i), "skipped": "no_pred"}

        try:
            device = pred_queries.device
            if not torch.is_tensor(gt_inst):
                gt_inst = torch.as_tensor(gt_inst)
            gt_inst = gt_inst.to(device=device)
            gt_inst = gt_inst.long().flatten()
            pred_masks = pred_masks.to(device=device)
            if pred_masks.dtype != torch.bool:
                pred_masks = pred_masks.bool()
            pred_queries = pred_queries.to(device=device)
        except Exception as e:
            return {"frame": int(frame_i), "skipped": "tensor_cast", "err": repr(e)}

        if pred_masks.dim() != 2 or pred_queries.dim() != 2:
            return {
                "frame": int(frame_i),
                "skipped": "bad_shapes",
                "pred_masks_shape": list(pred_masks.shape),
                "pred_queries_shape": list(pred_queries.shape),
            }

        n_pred = int(pred_masks.shape[0])
        n_pts = int(pred_masks.shape[1])
        if int(pred_queries.shape[0]) != n_pred:
            return {
                "frame": int(frame_i),
                "skipped": "shape_mismatch",
                "n_pred": n_pred,
                "n_query": int(pred_queries.shape[0]),
            }
        if gt_inst.numel() != n_pts:
            return {
                "frame": int(frame_i),
                "skipped": "gt_pred_pts_mismatch",
                "n_pts_pred": n_pts,
                "n_pts_gt": int(gt_inst.numel()),
            }

        # Visible GT instances (exclude -1) with enough points.
        try:
            ids, counts = torch.unique(gt_inst, return_counts=True)
            keep = ids != -1
            ids = ids[keep]
            counts = counts[keep]
            if gt_vis_npoint > 0:
                keep = counts >= gt_vis_npoint
                ids = ids[keep]
                counts = counts[keep]
        except Exception as e:
            return {"frame": int(frame_i), "skipped": "gt_unique_fail", "err": repr(e)}

        n_gt_vis = int(ids.numel())
        if n_gt_vis == 0:
            return {"frame": int(frame_i), "skipped": "no_visible_gt"}

        # Build GT masks matrix: [n_gt, n_pts] (bool)
        try:
            gt_masks = (gt_inst.unsqueeze(0) == ids.unsqueeze(1))
            gt_sizes = gt_masks.sum(dim=1).float()  # [n_gt]
        except Exception as e:
            return {"frame": int(frame_i), "skipped": "gt_mask_build_fail", "err": repr(e)}

        # Compute IoU between preds and GTs: [n_pred, n_gt]
        try:
            pred_f = pred_masks.float()
            gt_f = gt_masks.float()
            inter = pred_f @ gt_f.t()  # [n_pred, n_gt]
            pred_sz = pred_f.sum(dim=1, keepdim=True)  # [n_pred, 1]
            union = pred_sz + gt_sizes.unsqueeze(0) - inter
            iou = inter / (union + 1e-6)
            best_iou, best_pi = iou.max(dim=0)  # per-gt best pred idx
        except Exception as e:
            return {"frame": int(frame_i), "skipped": "iou_fail", "err": repr(e)}

        # Select GTs with a matched pred (min_iou).
        matched = best_iou >= float(min_iou)
        n_gt_matched = int(matched.sum().item())
        if n_gt_matched == 0:
            # Still update state? No, keep previous.
            out = {
                "frame": int(frame_i),
                "n_gt_vis": n_gt_vis,
                "n_gt_matched": 0,
                "n_pred": n_pred,
                "min_iou": float(min_iou),
                "iou_thr": float(iou_thr),
            }
            out.update(self._quantile_stats(best_iou.detach(), prefix="gt_best_iou_"))
            return out

        gt_ids_mat = ids[matched]
        pi_mat = best_pi[matched]
        iou_mat = best_iou[matched].detach()

        # Get per-GT embedding from best-matching pred.
        try:
            emb = pred_queries[pi_mat]  # [n_gt_matched, D]
            emb = F.normalize(emb.float(), dim=1)
        except Exception as e:
            return {"frame": int(frame_i), "skipped": "emb_fail", "err": repr(e)}

        # Positive cosine vs previous frame embeddings (consecutive only).
        pos_cos = []
        pos_strong_cos = []
        prev = state.get("prev", {})
        new_prev = prev.copy() if isinstance(prev, dict) else {}
        for k in range(int(gt_ids_mat.numel())):
            gid = int(gt_ids_mat[k].item())
            e_cur = emb[k]
            rec = new_prev.get(gid, None)
            if isinstance(rec, dict):
                prev_f = int(rec.get("frame", -999999))
                e_prev = rec.get("emb", None)
                if prev_f == int(frame_i - frame_stride) and torch.is_tensor(e_prev):
                    try:
                        c = float((e_prev * e_cur).sum().clamp(-1, 1).item())
                        pos_cos.append(c)
                        if float(iou_mat[k].item()) >= float(iou_thr):
                            pos_strong_cos.append(c)
                    except Exception:
                        pass
            new_prev[gid] = {"frame": int(frame_i), "emb": e_cur.detach()}
        state["prev"] = new_prev

        # Negative cosine within this frame (between different GT ids).
        neg_vals = torch.empty(0, device=device, dtype=torch.float32)
        try:
            n = int(emb.shape[0])
            if n >= 2:
                sim = emb @ emb.t()
                mask = ~torch.eye(n, device=device, dtype=torch.bool)
                vals = sim[mask].flatten()
                if vals.numel() > 0 and neg_pairs > 0 and vals.numel() > neg_pairs:
                    idx = torch.randperm(vals.numel(), device=device)[:neg_pairs]
                    vals = vals[idx]
                neg_vals = vals.detach()
        except Exception:
            neg_vals = torch.empty(0, device=device, dtype=torch.float32)

        out = {
            "frame": int(frame_i),
            "n_gt_vis": n_gt_vis,
            "n_gt_matched": n_gt_matched,
            "n_pred": n_pred,
            "min_iou": float(min_iou),
            "iou_thr": float(iou_thr),
            "gt_best_iou_ge_thr_rate": float((best_iou >= float(iou_thr)).float().mean().item()),
        }
        out.update(self._quantile_stats(best_iou.detach(), prefix="gt_best_iou_"))
        out.update(self._quantile_stats(torch.as_tensor(pos_cos, device=device), prefix="pos_cos_", qs=(0.1, 0.5, 0.9)))
        out.update(self._quantile_stats(torch.as_tensor(pos_strong_cos, device=device), prefix="pos_strong_cos_", qs=(0.1, 0.5, 0.9)))
        out.update(self._quantile_stats(neg_vals, prefix="neg_cos_", qs=(0.5, 0.9, 0.99)))
        out["sep_pos50_neg90"] = float(out.get("pos_cos_p50", 0.0) - out.get("neg_cos_p90", 0.0))
        out["sep_strong_pos50_neg90"] = float(out.get("pos_strong_cos_p50", 0.0) - out.get("neg_cos_p90", 0.0))
        if save_raw:
            try:
                out["pos_raw"] = [float(x) for x in pos_cos[:max_raw]]
            except Exception:
                out["pos_raw"] = []
            try:
                out["pos_strong_raw"] = [float(x) for x in pos_strong_cos[:max_raw]]
            except Exception:
                out["pos_strong_raw"] = []
            try:
                if neg_vals.numel() > 0:
                    _neg = neg_vals
                    if max_raw > 0 and _neg.numel() > max_raw:
                        _neg = _neg[:max_raw]
                    out["neg_raw"] = [float(x) for x in _neg.detach().cpu().tolist()]
                else:
                    out["neg_raw"] = []
            except Exception:
                out["neg_raw"] = []
        return out

        frame_stride = int(cfg.get("frame_stride", 10))
        max_frames = int(cfg.get("max_frames", 0))
        if frame_stride > 1 and (int(frame_i) % frame_stride) != 0:
            return {"skipped": "stride", "frame": int(frame_i)}
        seen = int(getattr(self, "_gdino_diag_seen", 0))
        if max_frames > 0 and seen >= max_frames:
            return {"skipped": "max_frames", "frame": int(frame_i)}

        meta = self._safe_get_img_metas(batch_data_samples)
        cam_info = meta.get("cam_info", None)
        img_paths = meta.get("img_paths", None)
        if cam_info is None or img_paths is None:
            return {"skipped": "no_cam_meta", "frame": int(frame_i)}
        if isinstance(cam_info, dict):
            cam = cam_info
        elif isinstance(cam_info, list) and len(cam_info) > 0:
            cam = cam_info[frame_i] if frame_i < len(cam_info) else cam_info[0]
        else:
            return {"skipped": "bad_cam_info", "frame": int(frame_i)}
        if not isinstance(cam, dict):
            return {"skipped": "bad_cam_item", "frame": int(frame_i)}

        try:
            intr = cam.get("intrinsics", None)
            if not torch.is_tensor(intr):
                intr = torch.as_tensor(intr, dtype=torch.float32)
            intr = intr.reshape(-1)[:4].to(torch.float32)
            fx, fy, cx, cy = [float(x) for x in intr.tolist()]
        except Exception:
            return {"skipped": "bad_intrinsics", "frame": int(frame_i)}

        pose = cam.get("pose", None)
        if pose is None:
            pose = cam.get("extrinsics", None)
        if pose is None:
            return {"skipped": "no_pose", "frame": int(frame_i)}
        if not torch.is_tensor(pose):
            pose = torch.as_tensor(pose, dtype=torch.float32)
        pose = pose.reshape(4, 4).to(torch.float32)

        hw = cam.get("img_size_gdino", None)
        if torch.is_tensor(hw) and hw.numel() == 2:
            h_img, w_img = int(hw.reshape(-1)[0].item()), int(hw.reshape(-1)[1].item())
        else:
            ts = cfg.get("target_size", (420, 560))
            h_img, w_img = int(ts[0]), int(ts[1])

        # Resolve current frame image path.
        if isinstance(img_paths, list) and len(img_paths) > 0:
            img_path = img_paths[frame_i] if frame_i < len(img_paths) else img_paths[0]
        else:
            img_path = img_paths
        if not isinstance(img_path, str):
            return {"skipped": "bad_img_path", "frame": int(frame_i)}

        # Points in the same coordinate system as `pose` expects (ScanNet world).
        try:
            pts = batch_inputs_dict.get("points", None)
            if isinstance(pts, (list, tuple)):
                xyz_world = pts[0][frame_i, :, :3]
            else:
                xyz_world = pts[frame_i, :, :3]
            if not torch.is_tensor(xyz_world):
                return {"skipped": "no_points", "frame": int(frame_i)}
        except Exception:
            return {"skipped": "no_points", "frame": int(frame_i)}

        device = xyz_world.device
        pose = pose.to(device=device)
        xyz_world = xyz_world.to(device=device)

        # Load and resize image online (deterministic).
        try:
            img = Image.open(img_path).convert("RGB")
            if img.size != (w_img, h_img):
                img = img.resize((w_img, h_img), resample=Image.BILINEAR)
            # np.asarray(PIL.Image) may return a non-writable view; copy to avoid undefined behavior warning.
            img_t = torch.from_numpy(np.asarray(img).copy()).to(device=device).float() / 255.0
            img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        except Exception:
            return {"skipped": "img_load_failed", "frame": int(frame_i)}

        # Build GDINO feature maps (backbone-only by default).
        try:
            gdino = self._get_gdino_backbone()
            out = gdino(img_t, backbone_only=bool(cfg.get("backbone_only", True)))
            srcs = out.get("srcs", None)
            if not isinstance(srcs, list) or len(srcs) == 0:
                return {"skipped": "no_srcs", "frame": int(frame_i)}
            level = int(cfg.get("feat_level", 0))
            level = max(0, min(level, len(srcs) - 1))
            feat_map = srcs[level]
            feat_h, feat_w = int(feat_map.shape[-2]), int(feat_map.shape[-1])
        except Exception:
            return {"skipped": "gdino_failed", "frame": int(frame_i)}

        max_depth = float(cfg.get("max_depth", 10.0))
        align_corners = bool(cfg.get("align_corners", False))

        def _project_ratio(mode: str) -> float:
            if mode == "inv":
                mat = torch.linalg.inv(pose)
            elif mode == "direct":
                mat = pose
            else:
                mat = torch.eye(4, device=device, dtype=xyz_world.dtype)
            xyz1 = torch.cat([xyz_world, torch.ones((xyz_world.shape[0], 1), device=device, dtype=xyz_world.dtype)], dim=1)
            xyz_cam = (xyz1 @ mat.T)[:, :3]
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            valid_z = (z > float(MIN_DEPTH)) & (z < max_depth)
            denom_ok = valid_z & (torch.abs(z) > torch.finfo(z.dtype).eps)
            ratio_x = torch.zeros_like(x)
            ratio_y = torch.zeros_like(y)
            if denom_ok.any():
                ratio_x[denom_ok] = x[denom_ok] / z[denom_ok]
                ratio_y[denom_ok] = y[denom_ok] / z[denom_ok]
            u_img = fx * ratio_x + cx
            v_img = fy * ratio_y + cy
            uv_img = torch.stack([u_img, v_img], dim=-1)
            uv_feat = scale_uv_img_to_feat(
                uv_img, img_hw=(h_img, w_img), feat_hw=(feat_h, feat_w), align_corners=align_corners
            )
            valid_u = (uv_feat[:, 0] >= 0) & (uv_feat[:, 0] < float(feat_w))
            valid_v = (uv_feat[:, 1] >= 0) & (uv_feat[:, 1] < float(feat_h))
            valid = valid_z & valid_u & valid_v
            return float(valid.float().mean().item())

        best_mode = "inv"
        best_ratio = _project_ratio("inv")
        for m in ("direct", "identity"):
            r = _project_ratio(m)
            if r > best_ratio:
                best_ratio, best_mode = r, m

        self._gdino_diag_seen = seen + 1
        return {
            "frame": int(frame_i),
            "img_hw": [int(h_img), int(w_img)],
            "feat_hw": [int(feat_h), int(feat_w)],
            "feat_level": int(level),
            "align_corners": bool(align_corners),
            "max_depth": float(max_depth),
            "pose_mode": str(best_mode),
            "valid_ratio": float(best_ratio),
        }

    def _get_gdino_daca2d_cfg(self, *, is_train: bool) -> dict:
        """Get DACA-2D config from train/test cfg with safe fallback."""
        cfg = {}
        try:
            base = (self.train_cfg or {}) if is_train else (self.test_cfg or {})
            cfg = base.get("gdino_daca2d", {}) or {}
        except Exception:
            cfg = {}
        # Fallback: allow configuring only in test_cfg while debugging training.
        if not cfg:
            try:
                cfg = (self.test_cfg or {}).get("gdino_daca2d", {}) or {}
            except Exception:
                cfg = {}
        return cfg if isinstance(cfg, dict) else {}

    def _run_gdino_daca2d_for_frame(
        self, batch_inputs_dict, batch_data_samples, frame_i: int, *, is_train: bool = False
    ):
        """Run GDINO full forward and build 2D query feats/pos for DACA-2D (optional)."""
        cfg = self._get_gdino_daca2d_cfg(is_train=is_train)
        if not (isinstance(cfg, dict) and bool(cfg.get("enable", False))):
            return None, None, None
        log_fail = bool(cfg.get("log_fail", False))
        log_first = bool(cfg.get("log_first", False))

        def _log(msg: str):
            try:
                from mmengine.logging import MMLogger
                logger = MMLogger.get_current_instance()
                if logger is not None:
                    logger.info(msg)
                    return
            except Exception:
                pass
            print(msg)

        def _skip(reason: str, **kw):
            stats = {"skipped": str(reason), "frame": int(frame_i)}
            stats.update({k: (int(v) if isinstance(v, bool) else v) for k, v in kw.items()})
            if log_fail:
                _log(f"[GDINO][daca2d][skip] frame={int(frame_i)} reason={reason} extra={kw}")
            return None, None, stats

        frame_stride = int(cfg.get("frame_stride", 1))
        max_frames = int(cfg.get("max_frames", 0))
        if frame_stride > 1 and (int(frame_i) % frame_stride) != 0:
            return _skip("stride")
        seen = int(getattr(self, "_gdino_daca2d_seen", 0))
        if max_frames > 0 and seen >= max_frames:
            return _skip("max_frames")

        def _as_vec4(x):
            # Robust unwrap for cam_info fields which may be nested lists from dataset/pipeline/collate.
            while isinstance(x, (list, tuple)) and len(x) == 1:
                x = x[0]
            if torch.is_tensor(x):
                t = x
            elif isinstance(x, (list, tuple)):
                # e.g. [tensor([fx,fy,cx,cy])] already unwrapped above,
                # or [fx,fy,cx,cy], or [tensor(fx), tensor(fy), ...]
                if len(x) == 4 and all(torch.is_tensor(v) and v.numel() == 1 for v in x):
                    t = torch.stack([v.reshape(()) for v in x], dim=0)
                else:
                    t = torch.as_tensor(x, dtype=torch.float32)
            else:
                t = torch.as_tensor(x, dtype=torch.float32)
            return t.reshape(-1)[:4].to(torch.float32)

        def _as_mat44(x):
            while isinstance(x, (list, tuple)) and len(x) == 1:
                x = x[0]
            if torch.is_tensor(x):
                t = x
            else:
                t = torch.as_tensor(x, dtype=torch.float32)
            return t.reshape(4, 4).to(torch.float32)

        batched_cam_info = batch_inputs_dict.get("cam_info", None)
        batched_img_paths = batch_inputs_dict.get("img_paths", None)
        pts_aug_in = batch_inputs_dict.get("points", None)
        pts_raw_in = batch_inputs_dict.get("points_raw", None)
        # Some dataloaders / data_preprocessors may keep cam_info/img_paths only in
        # data_samples.metainfo (and not in batch_inputs_dict). We allow that and
        # only hard-require points here; per-sample cam_info/img_paths will be
        # resolved in the loop below.
        if pts_aug_in is None:
            return _skip(
                "missing_inputs",
                has_cam_info=batched_cam_info is not None,
                has_img_paths=batched_img_paths is not None,
                has_points=pts_aug_in is not None,
                has_points_raw=pts_raw_in is not None,
            )

        B = len(batch_data_samples)
        # Collect per-sample GDINO meta (intr/pose/img size).
        per_intr = []
        per_pose = []
        per_hw = []
        per_img_path = []
        per_xyz_aug = []
        per_xyz_proj = []

        ts = cfg.get("img_size", (420, 560))
        default_h, default_w = int(ts[0]), int(ts[1])
        max_depth = float(cfg.get("max_depth", 10.0))

        for b in range(B):
            # Prefer per-sample metainfo (avoids collate-induced nesting).
            cam_info_b = None
            img_paths_b = None
            try:
                meta_b = getattr(batch_data_samples[b], "img_metas", None)
                if isinstance(meta_b, dict):
                    cam_info_b = meta_b.get("cam_info", None)
                    img_paths_b = meta_b.get("img_paths", None)
                else:
                    meta_b = getattr(batch_data_samples[b], "metainfo", None)
                    if isinstance(meta_b, dict):
                        cam_info_b = meta_b.get("cam_info", None)
                        img_paths_b = meta_b.get("img_paths", None)
            except Exception:
                cam_info_b = None
                img_paths_b = None
            # Fallback to batched inputs if per-sample metainfo is missing.
            if cam_info_b is None and batched_cam_info is not None:
                try:
                    cam_info_b = batched_cam_info[b]
                except Exception:
                    pass
            if img_paths_b is None and batched_img_paths is not None:
                try:
                    img_paths_b = batched_img_paths[b]
                except Exception:
                    pass

            if cam_info_b is None and isinstance(batched_cam_info, list) and len(batched_cam_info) == B:
                cam_info_b = batched_cam_info[b]
            if img_paths_b is None and isinstance(batched_img_paths, list) and len(batched_img_paths) == B:
                img_paths_b = batched_img_paths[b]

            # cam_info_b should be list[dict] (len=T) for online format.
            cam_list = cam_info_b
            if isinstance(cam_list, dict):
                cam = cam_list
            elif isinstance(cam_list, list) and len(cam_list) > 0:
                cam = cam_list[frame_i] if frame_i < len(cam_list) else cam_list[0]
            else:
                cam = None
            if not isinstance(cam, dict):
                return _skip("bad_cam_info", sample=int(b))

            try:
                intr = _as_vec4(cam.get("intrinsics", None))
                fx, fy, cx, cy = [float(v) for v in intr.tolist()]
            except Exception:
                if log_fail:
                    try:
                        raw_intr = cam.get("intrinsics", None)
                        intr_type = type(raw_intr).__name__
                        intr_len = len(raw_intr) if isinstance(raw_intr, (list, tuple)) else None
                    except Exception:
                        intr_type, intr_len = None, None
                    return _skip("bad_intrinsics", sample=int(b), intr_type=intr_type, intr_len=intr_len)
                return _skip("bad_intrinsics", sample=int(b))

            pose = cam.get("pose", None)
            if pose is None:
                pose = cam.get("extrinsics", None)
            if pose is None:
                return _skip("no_pose", sample=int(b))
            pose = _as_mat44(pose)

            hw = cam.get("img_size_gdino", None)
            if torch.is_tensor(hw) and hw.numel() == 2:
                h_img, w_img = int(hw.reshape(-1)[0].item()), int(hw.reshape(-1)[1].item())
            else:
                h_img, w_img = default_h, default_w

            if isinstance(img_paths_b, list) and len(img_paths_b) > 0:
                img_path = img_paths_b[frame_i] if frame_i < len(img_paths_b) else img_paths_b[0]
            else:
                img_path = img_paths_b
            if not isinstance(img_path, str):
                return _skip("bad_img_path", sample=int(b))

            # Points: wo-elastic augmented (decoder distance space)
            if isinstance(pts_aug_in, (list, tuple)):
                xyz_aug = pts_aug_in[b][frame_i, :, :3]
            else:
                xyz_aug = pts_aug_in[frame_i, :, :3]
            if not torch.is_tensor(xyz_aug):
                return _skip("no_points", sample=int(b))

            # Projection space: prefer raw points captured pre-3D-aug (aligned with pose/intr).
            if pts_raw_in is not None:
                if isinstance(pts_raw_in, (list, tuple)):
                    xyz_proj = pts_raw_in[b][frame_i, :, :3]
                else:
                    xyz_proj = pts_raw_in[frame_i, :, :3]
            else:
                # Fallback: reverse 3D aug using metadata (same as point_fusion).
                try:
                    img_meta_b = batch_data_samples[b].img_metas if hasattr(batch_data_samples[b], "img_metas") else {}
                    xyz_proj = apply_3d_transformation(xyz_aug, "DEPTH", img_meta_b, reverse=True)
                except Exception:
                    xyz_proj = xyz_aug

            per_intr.append((fx, fy, cx, cy))
            per_pose.append(pose)
            per_hw.append((h_img, w_img))
            per_img_path.append(img_path)
            per_xyz_aug.append(xyz_aug)
            per_xyz_proj.append(xyz_proj)

        hs_last = pred_boxes = pred_scores = None
        cache_dir = cfg.get("cache_dir", None)
        if cache_dir is None and isinstance(cfg.get("cache", None), dict):
            cache_dir = cfg.get("cache", {}).get("dir", None) or cfg.get("cache", {}).get("cache_dir", None)
        if cache_dir is None:
            cache_dir = os.environ.get("GDINO_CACHE_DIR", None)

        if cache_dir is not None and per_img_path:
            target_hw = per_hw[0]
            if all(h == target_hw for h in per_hw):
                bb_cfg = cfg.get("backbone", None) if isinstance(cfg, dict) else None
                if not isinstance(bb_cfg, dict):
                    bb_cfg = getattr(self, "_gdino_backbone_cfg", None)
                cached = load_gdino_cache_batched(
                    cache_dir,
                    img_paths=per_img_path,
                    target_hw=target_hw,
                    bb_cfg=bb_cfg if isinstance(bb_cfg, dict) else None,
                    mode="full",
                    device=per_xyz_aug[0].device,
                    dtype=torch.float32,
                )
                if isinstance(cached, dict):
                    hs_last = cached.get("hs_last", None)
                    pred_boxes = cached.get("pred_boxes", None)
                    pred_scores = cached.get("pred_scores", None)

        if hs_last is None or pred_boxes is None or pred_scores is None:
            img_t_list = []
            for b in range(B):
                img_path = per_img_path[b]
                h_img, w_img = per_hw[b]
                device_b = per_xyz_aug[b].device
                try:
                    img = Image.open(img_path).convert("RGB")
                    if img.size != (w_img, h_img):
                        img = img.resize((w_img, h_img), resample=Image.BILINEAR)
                    img_t = torch.from_numpy(np.asarray(img).copy()).to(device=device_b).float() / 255.0
                    img_t = img_t.permute(2, 0, 1)
                except Exception:
                    return _skip("img_load_failed", sample=int(b))
                img_t_list.append(img_t)

            img_batch = torch.stack(img_t_list, dim=0)
            try:
                gdino = self._get_gdino_backbone(cfg)
                with torch.no_grad():
                    out = gdino(img_batch, backbone_only=False)
                hs_last = out.get("hs_last", None)
                pred_boxes = out.get("pred_boxes", None)
                pred_scores = out.get("pred_scores", None)
            except Exception:
                return _skip("gdino_failed")
            if hs_last is None or pred_boxes is None or pred_scores is None:
                return _skip("gdino_missing")

        # Per-sample build q_feats/q_pos.
        score_thr = float(cfg.get("score_thr", 0.05))
        max_queries = int(cfg.get("max_queries", 300))
        select_mode = str(cfg.get("select_mode", "score_thr")).lower()
        qpos_ds_cfg = cfg.get("qpos_downsample", {}) if isinstance(cfg, dict) else {}
        qpos_ds_voxel = float(qpos_ds_cfg.get("voxel_size", 0.3))
        qpos_ds_pre_topk = int(qpos_ds_cfg.get("pre_topk", max_queries * 4))
        min_support = int(cfg.get("query3d_center", {}).get("min_support_pts", 30))

        q_feats_list, q_pos_list = [], []
        valid_ratios = []
        nq_keeps = []
        nq_pos_list = []

        for b in range(B):
            fx, fy, cx, cy = per_intr[b]
            pose = per_pose[b].to(device=per_xyz_aug[b].device)
            xyz_proj = per_xyz_proj[b].to(device=per_xyz_aug[b].device)
            xyz_aug = per_xyz_aug[b].to(device=per_xyz_aug[b].device)
            h_img, w_img = per_hw[b]

            # Choose best pose mode based on valid ratio (robust across datasets).
            def _project_ratio(mode: str) -> float:
                if mode == "inv":
                    mat = torch.linalg.inv(pose)
                elif mode == "direct":
                    mat = pose
                else:
                    mat = torch.eye(4, device=xyz_proj.device, dtype=xyz_proj.dtype)
                xyz1 = torch.cat(
                    [xyz_proj, torch.ones((xyz_proj.shape[0], 1), device=xyz_proj.device, dtype=xyz_proj.dtype)],
                    dim=1,
                )
                xyz_cam = (xyz1 @ mat.T)[:, :3]
                x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
                valid_z = (z > float(MIN_DEPTH)) & (z < max_depth)
                denom_ok = valid_z & (torch.abs(z) > torch.finfo(z.dtype).eps)
                ratio_x = torch.zeros_like(x)
                ratio_y = torch.zeros_like(y)
                if denom_ok.any():
                    ratio_x[denom_ok] = x[denom_ok] / z[denom_ok]
                    ratio_y[denom_ok] = y[denom_ok] / z[denom_ok]
                u_img = fx * ratio_x + cx
                v_img = fy * ratio_y + cy
                valid_u = (u_img >= 0) & (u_img < float(w_img))
                valid_v = (v_img >= 0) & (v_img < float(h_img))
                valid = valid_z & valid_u & valid_v
                return float(valid.float().mean().item())

            best_mode = "inv"
            best_ratio = _project_ratio("inv")
            for m in ("direct", "identity"):
                r = _project_ratio(m)
                if r > best_ratio:
                    best_ratio, best_mode = r, m

            if best_mode == "inv":
                mat = torch.linalg.inv(pose)
            elif best_mode == "direct":
                mat = pose
            else:
                mat = torch.eye(4, device=xyz_proj.device, dtype=xyz_proj.dtype)

            xyz1 = torch.cat(
                [xyz_proj, torch.ones((xyz_proj.shape[0], 1), device=xyz_proj.device, dtype=xyz_proj.dtype)],
                dim=1,
            )
            xyz_cam = (xyz1 @ mat.T)[:, :3]
            uv_img, valid_mask = project_points_to_uv(
                xyz_cam,
                feat_hw=(h_img, w_img),
                max_depth=max_depth,
                standard_intrinsics=(fx, fy, cx, cy),
                already_scaled=True,
            )
            valid_ratios.append(float(best_ratio))

            hs_b = hs_last[b]
            boxes_b = pred_boxes[b]
            scores_b = pred_scores[b]
            nq_raw = int(boxes_b.shape[0])
            # ---- query selection (score_thr / fixed top-k / qpos coverage downsample) ----
            if nq_raw <= 0:
                keep_idx = scores_b.new_zeros((0,), dtype=torch.long)
            elif select_mode == "topk":
                k = min(max_queries, nq_raw)
                keep_idx = torch.topk(scores_b, k=k, largest=True).indices
            elif select_mode == "qpos_voxel":
                k = min(max(qpos_ds_pre_topk, max_queries), nq_raw)
                keep_idx = torch.topk(scores_b, k=k, largest=True).indices
            else:
                keep = scores_b >= score_thr
                if keep.any():
                    keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
                    if keep_idx.numel() > max_queries:
                        topk = torch.topk(scores_b[keep_idx], k=max_queries, largest=True).indices
                        keep_idx = keep_idx[topk]
                else:
                    keep_idx = scores_b.new_zeros((0,), dtype=torch.long)

            q_feats = hs_b[keep_idx]
            q_boxes = boxes_b[keep_idx]
            nq_keep = int(q_boxes.shape[0])
            nq_keeps.append(nq_keep)

            # Convert normalized boxes to pixel coords (image space).
            cx_b = q_boxes[:, 0] * float(w_img)
            cy_b = q_boxes[:, 1] * float(h_img)
            bw = q_boxes[:, 2] * float(w_img)
            bh = q_boxes[:, 3] * float(h_img)
            x1 = (cx_b - 0.5 * bw).clamp(min=0.0, max=float(w_img))
            y1 = (cy_b - 0.5 * bh).clamp(min=0.0, max=float(h_img))
            x2 = (cx_b + 0.5 * bw).clamp(min=0.0, max=float(w_img))
            y2 = (cy_b + 0.5 * bh).clamp(min=0.0, max=float(h_img))

            u = uv_img[:, 0]
            v = uv_img[:, 1]
            q_pos_keep = []
            q_feat_keep = []
            q_score_keep = []
            support_counts = []
            box_areas = []
            for j in range(nq_keep):
                in_box = (u >= x1[j]) & (u <= x2[j]) & (v >= y1[j]) & (v <= y2[j]) & valid_mask
                # IMPORTANT: support selection uses raw projection (uv from xyz_proj),
                # but q_pos must be in decoder distance space (wo-elastic xyz_aug).
                support_pts = xyz_aug[in_box]
                support_counts.append(int(support_pts.shape[0]))
                try:
                    box_areas.append(float(((x2[j] - x1[j]) * (y2[j] - y1[j])).item()))
                except Exception:
                    box_areas.append(0.0)
                if support_pts.shape[0] < min_support:
                    continue
                q_pos_keep.append(torch.median(support_pts, dim=0).values)
                q_feat_keep.append(q_feats[j])
                try:
                    q_score_keep.append(scores_b[keep_idx[j]])
                except Exception:
                    pass

            # Optional: qpos coverage downsample (SegDINO3D-style "provide O object-level features").
            # We first keep a pre-topk by score (done above), then voxel-grid downsample by qpos.
            if select_mode == "qpos_voxel" and len(q_pos_keep) > 0 and len(q_score_keep) == len(q_pos_keep):
                try:
                    q_pos_tmp = torch.stack(q_pos_keep, dim=0)
                    q_feat_tmp = torch.stack(q_feat_keep, dim=0)
                    q_score_tmp = torch.stack(q_score_keep, dim=0).float()
                    vox = torch.floor(q_pos_tmp / max(qpos_ds_voxel, 1e-6)).to(torch.int64)
                    uniq, inv = torch.unique(vox, dim=0, return_inverse=True)
                    pick = []
                    for u_id in range(int(uniq.shape[0])):
                        inds = torch.nonzero(inv == u_id, as_tuple=False).squeeze(-1)
                        if inds.numel() == 1:
                            pick.append(int(inds.item()))
                        else:
                            pick.append(int(inds[q_score_tmp[inds].argmax()].item()))
                    pick = torch.as_tensor(pick, device=q_pos_tmp.device, dtype=torch.long)
                    if pick.numel() > max_queries:
                        topk = torch.topk(q_score_tmp[pick], k=max_queries, largest=True).indices
                        pick = pick[topk]
                    order = torch.argsort(q_score_tmp[pick], descending=True)
                    pick = pick[order]
                    q_pos_keep = [q_pos_tmp[i] for i in pick.tolist()]
                    q_feat_keep = [q_feat_tmp[i] for i in pick.tolist()]
                    q_score_keep = [q_score_tmp[i] for i in pick.tolist()]
                except Exception:
                    pass

            if len(q_pos_keep) == 0:
                q_pos = xyz_aug.new_zeros((0, 3))
                q_feat = xyz_aug.new_zeros((0, int(hs_b.shape[-1])))
                nq_pos = 0
            else:
                q_pos = torch.stack(q_pos_keep, dim=0)
                q_feat = torch.stack(q_feat_keep, dim=0)
                nq_pos = int(q_pos.shape[0])
            nq_pos_list.append(nq_pos)

            q_feats_list.append(q_feat)
            q_pos_list.append(q_pos)

            # Aggregate support stats across batch for diagnostics.
            try:
                if "support_counts_all" not in locals():
                    support_counts_all = []
                    box_areas_all = []
                support_counts_all.extend([int(x) for x in support_counts])
                box_areas_all.extend([float(x) for x in box_areas])
            except Exception:
                pass

        # ---- stats (for monitor/diagnostics) ----
        qpos_rates = []
        for k, p in zip(nq_keeps, nq_pos_list):
            if int(k) <= 0:
                qpos_rates.append(0.0)
            else:
                qpos_rates.append(float(p) / float(k))

        def _pctl(arr, q):
            try:
                if not arr:
                    return 0.0
                return float(np.percentile(np.asarray(arr, dtype=np.float32), q))
            except Exception:
                return 0.0

        stats = {
            "frame": int(frame_i),
            "batch": int(B),
            "select_mode": str(select_mode),
            "valid_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
            "valid_ratio_min": float(np.min(valid_ratios)) if valid_ratios else 0.0,
            "nq_keep_mean": float(np.mean(nq_keeps)) if nq_keeps else 0.0,
            "nq_pos_mean": float(np.mean(nq_pos_list)) if nq_pos_list else 0.0,
            "query_pos_valid_rate_mean": float(
                (np.sum(nq_pos_list) / max(np.sum(nq_keeps), 1)) if nq_keeps else 0.0
            ),
            "qpos_rate_min": float(np.min(qpos_rates)) if qpos_rates else 0.0,
            "qpos_rate_p50": _pctl(qpos_rates, 50),
            "qpos_rate_p90": _pctl(qpos_rates, 90),
            "nq_keep_p50": _pctl(nq_keeps, 50),
            "nq_keep_p90": _pctl(nq_keeps, 90),
            "nq_pos_p50": _pctl(nq_pos_list, 50),
            "nq_pos_p90": _pctl(nq_pos_list, 90),
        }

        # ---- support stats (in-box support points per 2D box) ----
        # These diagnose whether nq_pos is limited by point sparsity, tiny boxes, or projection mismatch.
        try:
            support_cfg = cfg.get("support_stats", {}) if isinstance(cfg, dict) else {}
            big_box_thr = float(support_cfg.get("big_box_area_px", 10000.0))
        except Exception:
            big_box_thr = 10000.0

        try:
            sc = np.asarray(locals().get("support_counts_all", []), dtype=np.float32)
            ba = np.asarray(locals().get("box_areas_all", []), dtype=np.float32)
            if sc.size > 0:
                stats.update(
                    {
                        "support_p50": float(np.percentile(sc, 50)),
                        "support_p90": float(np.percentile(sc, 90)),
                        "support_lt_min_rate": float(np.mean(sc < float(min_support))),
                        "box_area_p50": float(np.percentile(ba, 50)) if ba.size > 0 else 0.0,
                        "box_area_p90": float(np.percentile(ba, 90)) if ba.size > 0 else 0.0,
                        "big_box_thr": float(big_box_thr),
                    }
                )
                if ba.size == sc.size and ba.size > 0:
                    big = ba >= float(big_box_thr)
                    if np.any(big):
                        stats["big_box_rate"] = float(np.mean(big))
                        stats["big_box_low_support_rate"] = float(
                            np.mean((sc < float(min_support)) & big)
                        )
                    else:
                        stats["big_box_rate"] = 0.0
                        stats["big_box_low_support_rate"] = 0.0
        except Exception:
            pass

        log_every = int(cfg.get("log_valid_every", 0))
        if (log_every > 0 and (seen % log_every == 0)) or (log_first and seen == 0):
            _log(
                f"[GDINO][daca2d][valid_ratio] frame={int(frame_i)} "
                f"mean={stats['valid_ratio_mean']:.4f} min={stats['valid_ratio_min']:.4f} "
                f"qpos_rate_mean={stats['query_pos_valid_rate_mean']:.4f} "
                f"qpos_rate_min={stats['qpos_rate_min']:.4f} "
                f"nq_keep_mean={stats['nq_keep_mean']:.1f} nq_pos_mean={stats['nq_pos_mean']:.1f} "
                f"support_p50={stats.get('support_p50', 0.0):.1f} support_p90={stats.get('support_p90', 0.0):.1f} "
                f"lt_min={stats.get('support_lt_min_rate', 0.0):.3f} "
                f"big_low={stats.get('big_box_low_support_rate', 0.0):.3f}"
            )
        self._gdino_daca2d_seen = seen + 1
        return q_feats_list, q_pos_list, stats

    def _run_gdino_point_fusion_for_frame(self, batch_inputs_dict, batch_data_samples, frame_i: int):
        """Build per-point 2D features from GDINO image-level features (srcs).

        This is SegDINO3D-style "early fusion": project 3D points into the
        current frame image, sample multi-level 2D feature map, then project to
        a small per-point embedding and concatenate to the 3D backbone input.

        Notes:
        - Fully gated by `self.gdino_point_fusion_cfg.enable`.
        - Uses GDINO backbone_only=True (no tokenizer/text encoder) for speed.
        - Reverses 3D augmentation via `apply_3d_transformation(reverse=True)`
          when metadata is available, to preserve 2D-3D alignment under 3D aug.
        """
        cfg = self.gdino_point_fusion_cfg or {}
        if not (isinstance(cfg, dict) and bool(cfg.get("enable", False))):
            return None, None
        if self._gdino_point_proj is None:
            return None, {"skipped": "no_proj", "frame": int(frame_i)}

        def _log(msg: str):
            try:
                from mmengine.logging import MMLogger
                logger = MMLogger.get_current_instance()
                if logger is not None:
                    logger.info(msg)
                    return
            except Exception:
                pass
            print(msg)

        frame_stride = int(cfg.get("frame_stride", 1))
        max_frames = int(cfg.get("max_frames", 0))
        if frame_stride > 1 and (int(frame_i) % frame_stride) != 0:
            return None, {"skipped": "stride", "frame": int(frame_i)}
        seen = int(getattr(self, "_gdino_point_seen", 0))
        if max_frames > 0 and seen >= max_frames:
            return None, {"skipped": "max_frames", "frame": int(frame_i)}

        # Prefer per-sample metadata so batch_size>1 works.
        # Fallback to sample-0 metainfo for backward compatibility.
        batched_img_paths = batch_inputs_dict.get("img_paths", None)
        batched_cam_info = batch_inputs_dict.get("cam_info", None)

        feat_level = int(cfg.get("feat_level", 0))
        align_corners = bool(cfg.get("align_corners", False))
        max_depth = float(cfg.get("max_depth", 10.0))
        strict = bool(cfg.get("strict", False))
        strict_thr = float(cfg.get("strict_valid_ratio", 0.95))
        cache_dir = cfg.get("cache_dir", None)
        if cache_dir is None and isinstance(cfg.get("cache", None), dict):
            cache_dir = cfg.get("cache", {}).get("dir", None) or cfg.get("cache", {}).get("cache_dir", None)
        if cache_dir is None:
            cache_dir = os.environ.get("GDINO_CACHE_DIR", None)

        gdino_init_err = None
        try:
            gdino = self._get_gdino_backbone(cfg.get("gdino", None))
        except Exception as e:
            gdino = None
            gdino_init_err = repr(e)
        if gdino is None:
            err = {"skipped": "gdino_missing", "frame": int(frame_i)}
            if gdino_init_err is not None:
                err["gdino_init_error"] = gdino_init_err
            try:
                import torch as _torch
                err["cuda_available"] = bool(_torch.cuda.is_available())
                err["cuda_device_count"] = int(_torch.cuda.device_count())
            except Exception:
                pass
            if bool(cfg.get("log_fail", False)):
                _log(f"[GDINO][point_fusion][error] frame={int(frame_i)} err={err}")
            return None, err

        pts_raw_in = batch_inputs_dict.get("points_raw", None)
        pts_aug_in = batch_inputs_dict.get("points", None)
        if pts_raw_in is None and pts_aug_in is None:
            return None, {"skipped": "no_points", "frame": int(frame_i)}

        out_feats = []
        valid_ratios = []
        feat_h = feat_w = None
        last_best_mode = "na"
        for b in range(len(batch_data_samples)):
            # ---- cam_info (per-sample) ----
            cam_info_b = None
            try:
                meta_b = getattr(batch_data_samples[b], "img_metas", None)
                if isinstance(meta_b, dict):
                    cam_info_b = meta_b.get("cam_info", None)
                else:
                    meta_b = getattr(batch_data_samples[b], "metainfo", None)
                    if isinstance(meta_b, dict):
                        cam_info_b = meta_b.get("cam_info", None)
            except Exception:
                cam_info_b = None

            if cam_info_b is None and isinstance(batched_cam_info, list) and len(batched_cam_info) == len(batch_data_samples):
                cam_info_b = batched_cam_info[b]
            if cam_info_b is None:
                meta0 = self._safe_get_img_metas(batch_data_samples)
                cam_info_b = meta0.get("cam_info", None)

            if isinstance(cam_info_b, dict):
                cam_list = [cam_info_b]
            elif isinstance(cam_info_b, list) and len(cam_info_b) > 0:
                cam_list = cam_info_b
            else:
                out_feats.append(None)
                valid_ratios.append(0.0)
                continue

            cam = cam_list[frame_i] if frame_i < len(cam_list) else cam_list[0]
            if not isinstance(cam, dict):
                out_feats.append(None)
                valid_ratios.append(0.0)
                continue
            # intrinsics
            try:
                intr = cam.get("intrinsics", None)
                if torch.is_tensor(intr):
                    intr_t = intr.reshape(-1)[:4].to(torch.float32)
                elif isinstance(intr, (list, tuple)) and len(intr) == 1 and torch.is_tensor(intr[0]):
                    intr_t = intr[0].reshape(-1)[:4].to(torch.float32)
                else:
                    intr_t = torch.as_tensor(intr, dtype=torch.float32).reshape(-1)[:4]
                fx, fy, cx, cy = [float(v) for v in intr_t.tolist()]
            except Exception:
                out_feats.append(None)
                valid_ratios.append(0.0)
                continue

            pose = cam.get("pose", None)
            if pose is None:
                pose = cam.get("extrinsics", None)
            try:
                if torch.is_tensor(pose):
                    pose_t = pose.reshape(4, 4).to(torch.float32)
                elif isinstance(pose, (list, tuple)) and len(pose) == 1 and torch.is_tensor(pose[0]):
                    pose_t = pose[0].reshape(4, 4).to(torch.float32)
                else:
                    pose_t = torch.as_tensor(pose, dtype=torch.float32).reshape(4, 4)
            except Exception:
                pose_t = torch.eye(4, dtype=torch.float32)

            hw = cam.get("img_size_gdino", None)
            if torch.is_tensor(hw) and hw.numel() == 2:
                h_img, w_img = int(hw.reshape(-1)[0].item()), int(hw.reshape(-1)[1].item())
            else:
                ts = cfg.get("img_size", (420, 560))
                h_img, w_img = int(ts[0]), int(ts[1])

            # ---- img_path (per-sample) ----
            img_path = None
            if isinstance(batched_img_paths, list) and len(batched_img_paths) == len(batch_data_samples):
                try:
                    p = batched_img_paths[b]
                    if isinstance(p, list) and len(p) > 0:
                        img_path = p[frame_i] if frame_i < len(p) else p[0]
                    elif isinstance(p, str):
                        img_path = p
                except Exception:
                    img_path = None
            if img_path is None:
                try:
                    meta_b = getattr(batch_data_samples[b], "img_metas", None)
                    if isinstance(meta_b, dict):
                        ip = meta_b.get("img_paths", None)
                        if isinstance(ip, list) and len(ip) > 0:
                            img_path = ip[frame_i] if frame_i < len(ip) else ip[0]
                        elif isinstance(ip, str):
                            img_path = ip
                except Exception:
                    img_path = None
            if img_path is None:
                meta0 = self._safe_get_img_metas(batch_data_samples)
                ip = meta0.get("img_paths", None)
                if isinstance(ip, list) and len(ip) > 0:
                    img_path = ip[frame_i] if frame_i < len(ip) else ip[0]
                elif isinstance(ip, str):
                    img_path = ip
            if not isinstance(img_path, str):
                out_feats.append(None)
                valid_ratios.append(0.0)
                continue

            # points (aug xyz for downstream; raw xyz for projection when available)
            def _take_xyz(container, batch_idx: int):
                """Return (N,3) xyz for one sample at one frame.

                Supports:
                - list/tuple: container[batch_idx][frame_i]...
                - torch.Tensor: [B,T,N,C] or [T,N,C] or [N,C]
                - BasePoints-like objects exposing `.tensor`
                """
                if container is None:
                    return None
                x = None
                if isinstance(container, (list, tuple)):
                    x = container[batch_idx][frame_i, :, :3]
                elif torch.is_tensor(container):
                    if container.dim() == 4:
                        # [B, T, N, C]
                        x = container[batch_idx, frame_i, :, :3]
                    elif container.dim() == 3:
                        # [T, N, C]
                        x = container[frame_i, :, :3]
                    elif container.dim() == 2:
                        # [N, C]
                        x = container[:, :3]
                else:
                    # Fallback: try best-effort indexing, then unwrap `.tensor` if present.
                    try:
                        x = container[batch_idx][frame_i, :, :3]
                    except Exception:
                        try:
                            x = container[frame_i, :, :3]
                        except Exception:
                            x = container
                try:
                    if hasattr(x, "tensor"):
                        x = x.tensor
                except Exception:
                    pass
                if torch.is_tensor(x) and x.dim() > 2:
                    # Some pipelines may return [T,N,3] or [1,N,3]; flatten to [N,3].
                    x = x.reshape(-1, 3)
                return x

            xyz_aug = None
            try:
                xyz_aug = _take_xyz(pts_aug_in, b)
            except Exception:
                xyz_aug = None
            xyz_raw = None
            try:
                xyz_raw = _take_xyz(pts_raw_in, b)
            except Exception:
                xyz_raw = None

            if xyz_aug is None:
                xyz_aug = xyz_raw
            if xyz_aug is None or not torch.is_tensor(xyz_aug):
                out_feats.append(None)
                valid_ratios.append(0.0)
                continue

            device = xyz_aug.device
            pose_t = pose_t.to(device=device)

            # For 2D-3D alignment under 3D aug:
            # - If `points_raw` is provided, project with raw xyz (camera geometry is defined there),
            #   and later attach sampled features back to the same point indices (aug points).
            # - Otherwise (no raw points), try to reverse the 3D augmentation on the augmented xyz.
            used_raw_xyz = False
            xyz = None
            if torch.is_tensor(xyz_raw) and xyz_raw.shape == xyz_aug.shape:
                xyz = xyz_raw
                used_raw_xyz = True

            # Reverse 3D augmentation before projection when possible (only when raw xyz is unavailable).
            try:
                img_meta_b = batch_data_samples[b].img_metas if hasattr(batch_data_samples[b], "img_metas") else {}
            except Exception:
                img_meta_b = {}
            try:
                if xyz is None:
                    xyz = apply_3d_transformation(xyz_aug, "DEPTH", img_meta_b, reverse=True)
            except Exception:
                if xyz is None:
                    xyz = xyz_aug

            feat_map = None
            feat_h = feat_w = None
            if cache_dir is not None:
                bb_cfg = cfg.get("backbone", None) if isinstance(cfg, dict) else None
                if not isinstance(bb_cfg, dict):
                    bb_cfg = getattr(self, "_gdino_backbone_cfg", None)
                cached = load_gdino_cache_single(
                    cache_dir,
                    img_path=img_path,
                    target_hw=(int(h_img), int(w_img)),
                    bb_cfg=bb_cfg if isinstance(bb_cfg, dict) else None,
                    mode="full",
                    device=device,
                    dtype=torch.float32,
                )
                if isinstance(cached, dict):
                    srcs = cached.get("srcs", None)
                    if isinstance(srcs, list) and len(srcs) > 0:
                        level = max(0, min(int(feat_level), len(srcs) - 1))
                        feat_map = srcs[level]
                        if torch.is_tensor(feat_map) and feat_map.dim() == 4:
                            feat_h = int(feat_map.shape[-2])
                            feat_w = int(feat_map.shape[-1])
                if feat_map is None:
                    cached = load_gdino_cache_single(
                        cache_dir,
                        img_path=img_path,
                        target_hw=(int(h_img), int(w_img)),
                        bb_cfg=bb_cfg if isinstance(bb_cfg, dict) else None,
                        mode="backbone",
                        device=device,
                        dtype=torch.float32,
                    )
                    if isinstance(cached, dict):
                        srcs = cached.get("srcs", None)
                        if isinstance(srcs, list) and len(srcs) > 0:
                            level = max(0, min(int(feat_level), len(srcs) - 1))
                            feat_map = srcs[level]
                            if torch.is_tensor(feat_map) and feat_map.dim() == 4:
                                feat_h = int(feat_map.shape[-2])
                                feat_w = int(feat_map.shape[-1])

            if feat_map is None:
                # Load image (resized to img_size_gdino; enforce as safety net).
                try:
                    img = Image.open(img_path).convert("RGB")
                    if img.size != (w_img, h_img):
                        img = img.resize((w_img, h_img), resample=Image.BILINEAR)
                    img_t = torch.from_numpy(np.array(img, copy=True)).to(device=device).float() / 255.0
                    img_t = img_t.permute(2, 0, 1).unsqueeze(0)
                except Exception:
                    out_feats.append(None)
                    valid_ratios.append(0.0)
                    continue

                # GDINO backbone_only forward -> srcs
                try:
                    out = gdino(img_t, backbone_only=True)
                    srcs = out.get("srcs", None)
                    if not isinstance(srcs, list) or len(srcs) == 0:
                        raise RuntimeError("no_srcs")
                    level = max(0, min(int(feat_level), len(srcs) - 1))
                    feat_map = srcs[level]
                    if feat_map.dim() != 4:
                        raise RuntimeError("bad_feat_map")
                    feat_h = int(feat_map.shape[-2])
                    feat_w = int(feat_map.shape[-1])
                except Exception:
                    out_feats.append(None)
                    valid_ratios.append(0.0)
                    continue

            # Auto-pick pose mode based on valid ratio on the feature grid.
            xyz_world = xyz
            best_ratio, best_mode = -1.0, "identity"
            best_uv_feat, best_valid = None, None
            mode_ratios = {}
            for mode in ("inv", "direct", "identity"):
                if mode == "inv":
                    mat = torch.linalg.inv(pose_t)
                elif mode == "direct":
                    mat = pose_t
                else:
                    mat = torch.eye(4, device=device, dtype=xyz_world.dtype)
                xyz1 = torch.cat(
                    [xyz_world, torch.ones((xyz_world.shape[0], 1), device=device, dtype=xyz_world.dtype)], dim=1
                )
                xyz_cam = (xyz1 @ mat.T)[:, :3]
                uv_img, valid_mask = project_points_to_uv(
                    xyz_cam,
                    feat_hw=(h_img, w_img),
                    max_depth=max_depth,
                    standard_intrinsics=(fx, fy, cx, cy),
                    already_scaled=True,
                )
                uv_feat = scale_uv_img_to_feat(
                    uv_img, img_hw=(h_img, w_img), feat_hw=(feat_h, feat_w), align_corners=align_corners
                )
                valid_u = (uv_feat[:, 0] >= 0) & (uv_feat[:, 0] < float(feat_w))
                valid_v = (uv_feat[:, 1] >= 0) & (uv_feat[:, 1] < float(feat_h))
                valid = valid_mask & valid_u & valid_v
                ratio = float(valid.float().mean().item()) if valid.numel() else 0.0
                mode_ratios[mode] = ratio
                if ratio > best_ratio:
                    best_ratio, best_mode = ratio, mode
                    best_uv_feat, best_valid = uv_feat, valid

            # Fallback: if all modes fail, try raw (pre-reverse) identity.
            if best_ratio <= 0.0 and torch.is_tensor(xyz_raw):
                try:
                    xyz1 = torch.cat(
                        [xyz_raw, torch.ones((xyz_raw.shape[0], 1), device=device, dtype=xyz_raw.dtype)], dim=1
                    )
                    xyz_cam = xyz1[:, :3]
                    uv_img, valid_mask = project_points_to_uv(
                        xyz_cam,
                        feat_hw=(h_img, w_img),
                        max_depth=max_depth,
                        standard_intrinsics=(fx, fy, cx, cy),
                        already_scaled=True,
                    )
                    uv_feat = scale_uv_img_to_feat(
                        uv_img, img_hw=(h_img, w_img), feat_hw=(feat_h, feat_w), align_corners=align_corners
                    )
                    valid_u = (uv_feat[:, 0] >= 0) & (uv_feat[:, 0] < float(feat_w))
                    valid_v = (uv_feat[:, 1] >= 0) & (uv_feat[:, 1] < float(feat_h))
                    valid = valid_mask & valid_u & valid_v
                    ratio = float(valid.float().mean().item()) if valid.numel() else 0.0
                    if ratio > best_ratio:
                        best_ratio, best_mode = ratio, "raw_identity"
                        best_uv_feat, best_valid = uv_feat, valid
                        mode_ratios["raw_identity"] = ratio
                except Exception:
                    pass

            valid_ratios.append(float(best_ratio))
            last_best_mode = str(best_mode)
            if strict and best_ratio < strict_thr:
                raise RuntimeError(
                    f"[GDINO][point_fusion][strict] valid_ratio={best_ratio:.4f} < {strict_thr:.2f} "
                    f"sample={b} img={img_path} mode={best_mode} "
                    f"used_raw={int(used_raw_xyz)} ratios={mode_ratios}"
                )

            # Sample point-wise 2D feature and project to a small embedding.
            try:
                pt_feat = sample_img_feat(
                    feat_map, best_uv_feat, best_valid, align_corners=align_corners
                )
                pt_feat = self._gdino_point_proj(pt_feat)
                out_feats.append(pt_feat)
            except Exception as e:
                out_feats.append(None)
                if bool(cfg.get("log_fail", False)):
                    try:
                        fm_shape = tuple(feat_map.shape) if torch.is_tensor(feat_map) else None
                    except Exception:
                        fm_shape = None
                    try:
                        uv_shape = tuple(best_uv_feat.shape) if torch.is_tensor(best_uv_feat) else None
                    except Exception:
                        uv_shape = None
                    try:
                        valid_shape = tuple(best_valid.shape) if torch.is_tensor(best_valid) else None
                    except Exception:
                        valid_shape = None
                    try:
                        from mmengine.logging import MMLogger
                        logger = MMLogger.get_current_instance()
                        if logger is not None:
                            logger.info(
                                f"[GDINO][point_fusion][sample_fail] frame={int(frame_i)} b={b} "
                                f"err={repr(e)} feat_map={fm_shape} uv_feat={uv_shape} valid={valid_shape} "
                                f"mode={best_mode} img={img_path}"
                            )
                            continue
                    except Exception:
                        pass
                    print(
                        f"[GDINO][point_fusion][sample_fail] frame={int(frame_i)} b={b} "
                        f"err={repr(e)} feat_map={fm_shape} uv_feat={uv_shape} valid={valid_shape} "
                        f"mode={best_mode} img={img_path}"
                    )

        # Fill failures with zeros (keep shapes consistent).
        final_feats = []
        try:
            out_dim = int(getattr(self._gdino_point_proj, "out_features"))
        except Exception:
            # Identity() has no out_features; trust cfg for output dim.
            try:
                out_dim = int(cfg.get("out_dim", cfg.get("in_dim", 256)))
            except Exception:
                out_dim = 256
        for b in range(len(batch_data_samples)):
            if out_feats[b] is None:
                try:
                    if isinstance(pts_aug_in, (list, tuple)):
                        n = int(pts_aug_in[b][frame_i].shape[0])
                        device = pts_aug_in[b][frame_i].device
                    elif torch.is_tensor(pts_aug_in):
                        if pts_aug_in.dim() == 4:
                            n = int(pts_aug_in[b, frame_i].shape[0])
                            device = pts_aug_in.device
                        elif pts_aug_in.dim() == 3:
                            n = int(pts_aug_in[frame_i].shape[0])
                            device = pts_aug_in.device
                        else:
                            n = int(pts_aug_in.shape[0])
                            device = pts_aug_in.device
                    else:
                        raise RuntimeError("unsupported points container")
                except Exception:
                    n = 0
                    device = torch.device("cpu")
                final_feats.append(torch.zeros((n, out_dim), device=device))
            else:
                final_feats.append(out_feats[b])

        stats = {
            "frame": int(frame_i),
            "valid_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
            "valid_ratio_min": float(np.min(valid_ratios)) if valid_ratios else 0.0,
            "pose_mode": str(last_best_mode),
            "feat_level": int(feat_level),
            "feat_hw": [int(feat_h or 0), int(feat_w or 0)],
        }
        log_every = int(cfg.get("log_valid_every", 0))
        if log_every > 0 and (seen % log_every == 0):
            _log(
                f"[GDINO][point_fusion][valid_ratio] frame={int(frame_i)} "
                f"mean={stats['valid_ratio_mean']:.4f} "
                f"min={stats['valid_ratio_min']:.4f} "
                f"mode={stats['pose_mode']}"
            )
        self._gdino_point_seen = seen + 1
        return final_feats, stats

    def extract_feat(self, batch_inputs_dict, batch_data_samples, frame_i, track_instances=None):
        """Extract features from sparse tensor.
        """
        # construct tensor field
        coordinates, features = [], []
        coordinates_wo_elastic = []

        # Optional: per-point 2D feature fusion into 3D backbone input (SegDINO3D-style early fusion).
        gdino_point_feats = None
        gdino_point_stats = None
        try:
            gdino_point_feats, gdino_point_stats = self._run_gdino_point_fusion_for_frame(
                batch_inputs_dict, batch_data_samples, int(frame_i)
            )
        except Exception as e:
            gdino_point_feats, gdino_point_stats = None, {"frame": int(frame_i), "error": repr(e)}
            if bool(self.gdino_point_fusion_cfg.get("log_fail", False)):
                print(f"[GDINO][point_fusion][error] frame={int(frame_i)} err={repr(e)}")
        # Keep latest stats for online_monitor (side-channel).
        try:
            self._last_gdino_point_fusion_stats = gdino_point_stats
        except Exception:
            pass

        use_gdino_fusion = bool(self.gdino_point_fusion_cfg.get("enable", False))

        # Point-fusion mode control: when using sparse-FPN injection, do NOT concat features to backbone input.
        fuse_mode = str(self.gdino_point_fusion_cfg.get("fuse_mode", "concat")).lower()
        pf_mode = str(self.gdino_point_fusion_cfg.get("mode", "early")).lower()
        if fuse_mode in ("fpn", "sparse_fpn"):
            pf_mode = "fpn"

        gdino_sparse_fpn = None
        if use_gdino_fusion and pf_mode == "fpn":
            ok = isinstance(gdino_point_feats, (list, tuple)) and len(gdino_point_feats) == len(batch_inputs_dict.get('points', []))
            if not ok:
                raise RuntimeError(
                    "[GDINO][point_fusion] enabled but no per-point features were produced. "
                    f"stats={gdino_point_stats}"
                )
            coords_list = []
            feats_list = []
            for b in range(len(batch_inputs_dict.get('points', []))):
                if 'elastic_coords' in batch_inputs_dict:
                    coord_src = batch_inputs_dict['elastic_coords'][b][frame_i] * self.voxel_size
                else:
                    coord_src = batch_inputs_dict['points'][b][frame_i, :, :3]
                coords = torch.floor(coord_src / self.voxel_size).to(dtype=torch.int32)
                batch_col = torch.full((coords.shape[0], 1), b, dtype=torch.int32, device=coords.device)
                coords_batched = torch.cat([batch_col, coords], dim=1)
                coords_list.append(coords_batched)
                feats_list.append(gdino_point_feats[b].to(device=coords.device))
            coords_batch = torch.cat(coords_list, dim=0) if coords_list else None
            feats_batch = torch.cat(feats_list, dim=0) if feats_list else None
            if coords_batch is not None and feats_batch is not None:
                if coords_batch.shape[0] != feats_batch.shape[0]:
                    shapes = []
                    for b in range(len(batch_inputs_dict.get("points", []))):
                        n_coords = int(coords_list[b].shape[0]) if b < len(coords_list) else -1
                        fb = gdino_point_feats[b] if isinstance(gdino_point_feats, (list, tuple)) and b < len(gdino_point_feats) else None
                        n_feats = int(getattr(fb, "shape", [0])[0]) if torch.is_tensor(fb) else -1
                        shapes.append((b, n_coords, n_feats))
                    raise RuntimeError(
                        "[GDINO][point_fusion] merge_superpixels_train coords/feats length mismatch: "
                        f"coords_batch={tuple(coords_batch.shape)} feats_batch={tuple(feats_batch.shape)} "
                        f"per_sample={shapes}"
                    )
                gdino_sparse_fpn = build_sparse_fpn(coords_batch, feats_batch)
        fusion_out_dim = 0
        if use_gdino_fusion:
            if self._gdino_point_proj is not None:
                fusion_out_dim = int(self.gdino_point_fusion_cfg.get("out_dim", 0))
            else:
                fusion_out_dim = int(self.gdino_point_fusion_cfg.get("out_dim", 0))

        # Ensure point feature dims match backbone input channels (robust to cfg mismatch).
        expected_in = None
        try:
            if hasattr(self.backbone, "in_channels"):
                expected_in = int(self.backbone.in_channels)
            elif hasattr(self.backbone, "conv0p1s1") and hasattr(self.backbone.conv0p1s1, "in_channels"):
                expected_in = int(self.backbone.conv0p1s1.in_channels)
            elif hasattr(self.backbone, "conv0p1s1") and hasattr(self.backbone.conv0p1s1, "kernel"):
                expected_in = int(self.backbone.conv0p1s1.kernel.size(1))
        except Exception:
            expected_in = None

        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict: # False
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            # Always keep a wo-elastic coordinate copy for DACA-2D geometry (pos_wo_elastic).
            try:
                coordinates_wo_elastic.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            except Exception:
                coordinates_wo_elastic.append(coordinates[-1])
            rgb = batch_inputs_dict['points'][i][frame_i, :, 3:]
            if use_gdino_fusion and fusion_out_dim > 0 and pf_mode != "fpn":
                if gdino_point_feats is not None and i < len(gdino_point_feats) and gdino_point_feats[i] is not None:
                    try:
                        rgb = torch.cat(
                            [rgb, gdino_point_feats[i].to(device=rgb.device, dtype=rgb.dtype)],
                            dim=1,
                        )
                    except Exception:
                        rgb = torch.cat(
                            [rgb, torch.zeros((rgb.shape[0], fusion_out_dim), device=rgb.device, dtype=rgb.dtype)],
                            dim=1,
                        )
                else:
                    rgb = torch.cat(
                        [rgb, torch.zeros((rgb.shape[0], fusion_out_dim), device=rgb.device, dtype=rgb.dtype)],
                        dim=1,
                    )
            # Pad/trim to match backbone expected input channels.
            if expected_in is not None and rgb.shape[1] != expected_in:
                if not getattr(self, "_warned_in_channels_mismatch", False):
                    import warnings
                    warnings.warn(
                        f"[Backbone][in_channels] mismatch: got {rgb.shape[1]} but expected {expected_in}. "
                        f"gdino_fusion={use_gdino_fusion} out_dim={fusion_out_dim}. "
                        "Padding/truncating features to proceed."
                    )
                    self._warned_in_channels_mismatch = True
                if rgb.shape[1] < expected_in:
                    pad = torch.zeros(
                        (rgb.shape[0], expected_in - rgb.shape[1]),
                        device=rgb.device,
                        dtype=rgb.dtype,
                    )
                    rgb = torch.cat([rgb, pad], dim=1)
                else:
                    rgb = rgb[:, :expected_in]
            features.append(rgb)
        all_xyz = coordinates # [20000, 3]
        all_xyz_wo_elastic = coordinates_wo_elastic

        coordinates, features = ME.utils.batch_sparse_collate( # [20000, 4] [20000, 3]
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features) 

        # forward of backbone and neck
        if gdino_sparse_fpn is not None:
            try:
                x = self.backbone(field.sparse(), dino_feats=gdino_sparse_fpn, memory=self.memory if hasattr(self,'memory') else None)
            except TypeError:
                x = self.backbone(field.sparse(), dino_feats=gdino_sparse_fpn)
        else:
            x = self.backbone(field.sparse(), memory=self.memory if hasattr(self,'memory') else None) # [13141, 96]
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field) # [20000, 96]
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)] # [20000, 99] 坐标和特征进行拼接
        x = x.features # [20000, 96]

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        compact_sp_ids = []
        if self.use_temporal_loss and self.inst_dict is not None:
            best_obj_ids_list = []
        for batch_idx, (data_sample, tmp_xyz) in enumerate(zip(batch_data_samples, all_xyz)):
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i].clone() # [20000] 每个点属于的segment ID
            sp_pts_mask = sp_pts_mask.to(dtype=torch.long)
            _, sp_pts_mask = torch.unique(sp_pts_mask, sorted=True, return_inverse=True)
            compact_sp_ids.append(sp_pts_mask)
            if self.use_temporal_loss and self.inst_dict is not None:
                points_xyz = all_xyz[batch_idx]  # (N,3)
                point_ids  = sp_pts_mask # (N,)dd
                if 'bboxes_3d' not in self.inst_dict or batch_idx >= len(self.inst_dict.get('bboxes_3d', [])):
                    if getattr(self, "_log_missing_bboxes", False) and not getattr(self, "_warned_missing_bboxes", False):
                        import warnings
                        warnings.warn(
                            f"[WARN] missing gt bboxes_3d in train sample batch={batch_idx} frame={frame_i}"
                        )
                        self._warned_missing_bboxes = True
                    bboxes_6d = None
                else:
                    bboxes_6d  = self.inst_dict['bboxes_3d'][batch_idx]  # (M,6)

            sp_pts_masks.append(sp_pts_mask + sum(n_super_points)) # [20000] 每个点在所有点中的ID
            n_super_points.append(int(sp_pts_mask.max().item()) + 1 if sp_pts_mask.numel() else 0)
        sp_idx = torch.cat(sp_pts_masks) # [20000]
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz, with_xyz=True) # [N_segment, 96], [20000, 1]

        # Precompute SP positions in wo-elastic space for DACA-2D distance gating.
        # This must match decoder distance space (pos_wo_elastic), not elastic coords.
        sp_pos_wo_list = []
        try:
            for batch_idx, data_sample in enumerate(batch_data_samples):
                sp_id = compact_sp_ids[batch_idx].to(all_xyz_wo_elastic[batch_idx].device)
                xyz_wo = all_xyz_wo_elastic[batch_idx]
                sp_pos_wo = scatter_mean(xyz_wo, sp_id, dim=0)
                sp_pos_wo_list.append(sp_pos_wo)
        except Exception:
            sp_pos_wo_list = None
        try:
            self._last_sp_pos_wo_elastic_list = sp_pos_wo_list
        except Exception:
            pass

        if self.use_temporal_loss and self.inst_dict is not None:
            self.inst_dict['best_obj_ids_list'] = best_obj_ids_list
        # apply cls_layer
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        return features, point_features, all_xyz_w, sp_xyz_list # [N_segment, 96], [20000, 99], [20000, 1], [N_segment, 3]

    def merge_superpixels_extract_feat(self, batch_inputs_dict, batch_data_samples, frame_i, overlap_threshold=0.99):
        """Merge superpixels and extract features once.

        This integrates feature extraction with superpixel merging to avoid
        recomputing backbone/neck. It first pools by current SPs to decide
        merges, updates masks, then pools again to produce final features.

        Returns: features, point_features, all_xyz_w, sp_xyz_list
        """
        # 1) image features (same as extract_feat) — support no-image cases
        use_images = ('img_paths' in batch_inputs_dict) and hasattr(self, 'img_backbone')
        if use_images:
            with torch.no_grad():
                img_features = []
                for img_paths in batch_inputs_dict['img_paths']:
                    img_features.append(self.img_backbone(img_paths[frame_i])[0])
            img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]
            for img_meta in img_metas:
                img_meta['depth2img'] = img_meta['depth2img'][frame_i]

        # 2) Optional: per-point GDINO features (A3/A4 sparse-FPN injection)
        gdino_point_feats = None
        gdino_point_stats = None
        try:
            gdino_point_feats, gdino_point_stats = self._run_gdino_point_fusion_for_frame(
                batch_inputs_dict, batch_data_samples, int(frame_i)
            )
        except Exception as e:
            gdino_point_feats, gdino_point_stats = None, {"frame": int(frame_i), "error": repr(e)}
            if bool(getattr(self, "gdino_point_fusion_cfg", {}).get("log_fail", False)):
                print(f"[GDINO][point_fusion][error] frame={int(frame_i)} err={repr(e)}")
        try:
            self._last_gdino_point_fusion_stats = gdino_point_stats
        except Exception:
            pass

        enable_pf = bool(getattr(self, "gdino_point_fusion_cfg", {}).get("enable", False))
        fuse_mode = str(getattr(self, "gdino_point_fusion_cfg", {}).get("fuse_mode", "concat")).lower()
        pf_mode = str(getattr(self, "gdino_point_fusion_cfg", {}).get("mode", "early")).lower()
        if fuse_mode in ("fpn", "sparse_fpn"):
            pf_mode = "fpn"

        gdino_sparse_fpn = None
        if enable_pf and pf_mode == "fpn":
            ok = isinstance(gdino_point_feats, (list, tuple)) and len(gdino_point_feats) == len(batch_inputs_dict.get("points", []))
            if not ok:
                raise RuntimeError(
                    "[GDINO][point_fusion] enabled but no per-point features were produced. "
                    f"stats={gdino_point_stats}"
                )
            coords_list, feats_list = [], []
            for b in range(len(batch_inputs_dict.get("points", []))):
                if "elastic_coords" in batch_inputs_dict and batch_inputs_dict["elastic_coords"] is not None:
                    coord_src = batch_inputs_dict["elastic_coords"][b][frame_i] * self.voxel_size
                else:
                    coord_src = batch_inputs_dict["points"][b][frame_i, :, :3]
                coords = torch.floor(coord_src / self.voxel_size).to(dtype=torch.int32)
                batch_col = torch.full((coords.shape[0], 1), b, dtype=torch.int32, device=coords.device)
                coords_list.append(torch.cat([batch_col, coords], dim=1))
                feats_list.append(gdino_point_feats[b].to(device=coords.device))
            coords_batch = torch.cat(coords_list, dim=0) if coords_list else None
            feats_batch = torch.cat(feats_list, dim=0) if feats_list else None
            if coords_batch is not None and feats_batch is not None:
                if coords_batch.shape[0] != feats_batch.shape[0]:
                    per_sample = []
                    for b in range(len(coords_list)):
                        fb = gdino_point_feats[b] if isinstance(gdino_point_feats, (list, tuple)) else None
                        per_sample.append((b, int(coords_list[b].shape[0]), int(getattr(fb, "shape", [0])[0]) if torch.is_tensor(fb) else -1))
                    raise RuntimeError(
                        "[GDINO][point_fusion] merge_superpixels_extract_feat coords/feats length mismatch: "
                        f"coords_batch={tuple(coords_batch.shape)} feats_batch={tuple(feats_batch.shape)} "
                        f"per_sample={per_sample}"
                    )
                gdino_sparse_fpn = build_sparse_fpn(coords_batch, feats_batch)

        # 3) construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates

        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # 4) backbone + neck (once)
        if use_images:
            # NOTE: image-backbone fusion path is kept for compatibility with FF variants.
            # For Res16UNet34C (default), `use_images` is typically False.
            x = self.backbone(
                field.sparse(),
                partial(self._f, img_features=img_features, img_metas=img_metas, img_shape=img_metas[0]['img_shape']),
                memory=self.memory if hasattr(self, 'memory') else None
            )
        else:
            if gdino_sparse_fpn is not None:
                try:
                    x = self.backbone(field.sparse(), dino_feats=gdino_sparse_fpn, memory=self.memory if hasattr(self, 'memory') else None)
                except TypeError:
                    x = self.backbone(field.sparse(), dino_feats=gdino_sparse_fpn)
            else:
                x = self.backbone(field.sparse(), memory=self.memory if hasattr(self, 'memory') else None)
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c, f], dim=-1) for c, f in zip(all_xyz, x.decomposed_features)]
        x = x.features

        # 4) pool with current superpixels to compute merging decisions
        sp_pts_masks, n_super_points = [], []
        current_sp_pts_mask = [bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples]
        for sp_pts_mask in current_sp_pts_mask:
            sp_pts_mask = sp_pts_mask.to(dtype=torch.long)
            _, sp_pts_mask = torch.unique(sp_pts_mask, sorted=True, return_inverse=True)
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(int(sp_pts_mask.max().item()) + 1 if sp_pts_mask.numel() else 0)
        sp_idx = torch.cat(sp_pts_masks)
        x_pooled, all_xyz_w_orig = self.pool(x, sp_idx, all_xyz, with_xyz=True)

        features_orig, sp_xyz_list_orig = [], []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features_orig.append(x_pooled[begin: end, :-3])
            sp_xyz_list_orig.append(x_pooled[begin: end, -3:])

        # predictor for per-SP detections (lightweight path)
        x_detach = [features_orig[i] for i in range(len(features_orig))]
        pred_bboxes, pred_cls_list, new_queries = [], [], []
        queries = self.decoder._get_queries(x_detach, len(current_sp_pts_mask))
        for i in range(len(queries)):
            norm_query = self.decoder.out_norm(queries[i])
            reg_final = self.decoder.out_reg(norm_query)
            reg_cls = self.decoder.out_cls(norm_query)
            reg_cls = reg_cls.softmax(1)
            reg_distance = torch.exp(reg_final[:, 3:6])
            pred_bbox = torch.cat([reg_final[:, :3], reg_distance], dim=1)
            pred_cls_list.append(reg_cls)
            pred_bboxes.append(pred_bbox)
            new_queries.append(norm_query)
        x_detach = [new_queries[i] for i in range(len(new_queries))]

        valid_sps_list = []
        for batch_idx in range(len(pred_bboxes)):
            labels = pred_cls_list[batch_idx].argmax(dim=1)
            bg = pred_cls_list[batch_idx].shape[1] - 1
            labels_mask = ((labels[:, None] == labels) & (labels[:, None] != bg) & (labels != bg)[:, None])
            det_bboxes = pred_bboxes[batch_idx].clone()
            det_bboxes[:, :3] += sp_xyz_list_orig[batch_idx][:, :3]
            pos_embedding = self.merge_box_trans(det_bboxes)
            obj_embedding1, obj_embedding2 = self.merge_query_inter(x_detach[batch_idx], x_detach[batch_idx], pos_embedding)
            merge_rel_dist = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes))
            merge_rel_dist = merge_rel_dist.unsqueeze(-1)
            merge_geometry_embedding = self.merge_dist_embed(merge_rel_dist)
            merge_appear_embedding = obj_embedding1[:, None] * obj_embedding2[None]
            merge_fused_embedding = self.merge_fusion(merge_appear_embedding, merge_geometry_embedding)
            merge_det_mat = self.merge_embed_trans(merge_fused_embedding).sum(-1)
            m = merge_det_mat.sigmoid()
            iou_map = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes), mode='iou')
            cluster_both = cluster_complete_link(iou_map * m * labels_mask.float(), 0.5)
            valid_sps_list.append(cluster_both)

        # Optional: monitoring for LMI/SCL superpoint merging (fragment merge).
        # This is a pure side-channel: it must not affect merging behavior.
        monitor_cfg = {}
        try:
            monitor_cfg = (self.test_cfg.get('online_monitor', None) or {}) if hasattr(self, 'test_cfg') else {}
        except Exception:
            monitor_cfg = {}
        monitor_enable = bool(monitor_cfg.get('enable', False))
        sp_merge_stats = None
        if monitor_enable:
            try:
                # batch size is usually 1 at test time; keep per-batch stats list for robustness.
                per_batch = []
                for batch_idx in range(len(current_sp_pts_mask)):
                    sp_pts_mask0 = current_sp_pts_mask[batch_idx]
                    sp_before = int(sp_pts_mask0.max().item() + 1) if sp_pts_mask0.numel() else 0
                    gt_inst = None
                    try:
                        gt_inst = batch_data_samples[batch_idx].gt_pts_seg.pts_instance_mask[frame_i]
                    except Exception:
                        gt_inst = None
                    per_batch.append({
                        "sp_before": sp_before,
                        "gt_inst_available": bool(gt_inst is not None),
                    })
                sp_merge_stats = {
                    "frame": int(frame_i),
                    "merge_algo": "cluster_complete_link(iou_map*m*labels_mask, thr=0.5)",
                    "use_bbox": bool(getattr(self, 'use_bbox', False)),
                    "batch": per_batch,
                }
            except Exception:
                sp_merge_stats = {"frame": int(frame_i), "error": "sp_merge_stats_init_failed"}

        # generate merged masks and update in-place
        for batch_idx in range(len(current_sp_pts_mask)):
            sp_pts_mask = current_sp_pts_mask[batch_idx]
            merged_mask = sp_pts_mask.clone()
            merged_groups = []
            for valid_sps in valid_sps_list[batch_idx]:
                if len(valid_sps) > 1:
                    merged_groups.append(valid_sps)
                    for sp_id in valid_sps:
                        merged_mask[sp_pts_mask == sp_id] = valid_sps[0]
            all_original_ids = sp_pts_mask.unique().tolist()
            merged_ids = set()
            for group in merged_groups:
                merged_ids.update(group)
            unmerged_ids = [id for id in all_original_ids if id not in merged_ids]
            new_id = 0
            id_mapping = {}
            for group in merged_groups:
                for old_id in group:
                    id_mapping[old_id] = new_id
                new_id += 1
            for old_id in unmerged_ids:
                id_mapping[old_id] = new_id
                new_id += 1
            final_mask = merged_mask.clone()
            for old_id, nid in id_mapping.items():
                final_mask[merged_mask == old_id] = nid
            batch_data_samples[batch_idx].gt_pts_seg.sp_pts_mask[frame_i] = final_mask

            if monitor_enable and sp_merge_stats is not None:
                try:
                    sp_after = int(final_mask.max().item() + 1) if final_mask.numel() else 0
                    group_sizes = [int(len(g)) for g in merged_groups if isinstance(g, (list, tuple))]

                    # Pairwise "precision" of merges using per-frame GT instance ids (positive if same GT).
                    # This evaluates: among merged SP pairs, how many belong to the same GT instance.
                    # We do NOT attempt recall here (requires enumerating all positive pairs).
                    pair_total = pair_pos = pair_neg = pair_bg = 0
                    try:
                        gt_inst = batch_data_samples[batch_idx].gt_pts_seg.pts_instance_mask[frame_i]
                    except Exception:
                        gt_inst = None
                    if gt_inst is not None and gt_inst.numel() == sp_pts_mask.numel():
                        def _dominant_gt_id(sp_id: int) -> int:
                            pts = gt_inst[sp_pts_mask == sp_id]
                            pts = pts[pts >= 0]
                            if pts.numel() == 0:
                                return -1
                            vals, cnts = pts.unique(return_counts=True)
                            return int(vals[cnts.argmax()].item())

                        # Cache dominant ids to avoid repeated scans.
                        dom_cache = {}
                        for g in merged_groups:
                            if not isinstance(g, (list, tuple)) or len(g) < 2:
                                continue
                            dom_ids = []
                            for sp_id in g:
                                sp_id_int = int(sp_id)
                                if sp_id_int not in dom_cache:
                                    dom_cache[sp_id_int] = _dominant_gt_id(sp_id_int)
                                dom_ids.append(dom_cache[sp_id_int])
                            # Count pairs inside this group
                            n = len(dom_ids)
                            for i in range(n):
                                for j in range(i + 1, n):
                                    a, b = dom_ids[i], dom_ids[j]
                                    pair_total += 1
                                    if a < 0 or b < 0:
                                        pair_bg += 1
                                    elif a == b:
                                        pair_pos += 1
                                    else:
                                        pair_neg += 1

                    sp_merge_stats["batch"][batch_idx].update({
                        "sp_after": sp_after,
                        "merge_drop": int(sp_merge_stats["batch"][batch_idx].get("sp_before", 0) - sp_after),
                        "num_groups": int(len(merged_groups)),
                        "group_size": {
                            "n": int(len(group_sizes)),
                            "mean": float(sum(group_sizes) / max(len(group_sizes), 1)),
                            "max": int(max(group_sizes) if group_sizes else 0),
                        },
                        "pair": {
                            "total": int(pair_total),
                            "pos": int(pair_pos),
                            "neg": int(pair_neg),
                            "bg": int(pair_bg),
                            "pos_rate": float(pair_pos / max(pair_total, 1)),
                            "neg_rate": float(pair_neg / max(pair_total, 1)),
                        },
                    })
                except Exception:
                    sp_merge_stats["batch"][batch_idx].update({"error": "sp_merge_stats_failed"})

        # 5) pool with merged superpixels to produce final features
        sp_pts_masks_new, n_super_points_new = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i]
            sp_pts_mask = sp_pts_mask.to(dtype=torch.long)
            _, sp_pts_mask = torch.unique(sp_pts_mask, sorted=True, return_inverse=True)
            sp_pts_masks_new.append(sp_pts_mask + sum(n_super_points_new))
            n_super_points_new.append(int(sp_pts_mask.max().item()) + 1 if sp_pts_mask.numel() else 0)
        sp_idx_new = torch.cat(sp_pts_masks_new)
        x_pooled_new, all_xyz_w = self.pool(x, sp_idx_new, all_xyz, with_xyz=True)
        features_final, sp_xyz_list = [], []
        for i in range(len(n_super_points_new)):
            begin = sum(n_super_points_new[:i])
            end = sum(n_super_points_new[:i + 1])
            features_final.append(x_pooled_new[begin: end, :-3])
            sp_xyz_list.append(x_pooled_new[begin: end, -3:])
        if monitor_enable:
            # Store last frame stats for the outer predict() loop to consume.
            # Keep it minimal and JSON-friendly.
            try:
                self._last_sp_merge_stats = sp_merge_stats
            except Exception:
                pass
        return features_final, point_features, all_xyz_w, sp_xyz_list
    
    def _select_queries(self, x, gt_instances, sp_xyz, frame_i):
        """Select queries for train pass.
        """

        gt_instances_ = []
        for i in range(len(x)): # batch_size
            temp = InstanceData()
            temp.labels_3d = gt_instances[i].labels_3d[frame_i].to(x[i].device)
            temp.sp_masks = gt_instances[i].sp_masks[frame_i].to(x[i].device)
            n_labels = int(temp.labels_3d.shape[0])
            if hasattr(gt_instances[i], "bboxes_3d"):
                bboxes_3d = gt_instances[i].bboxes_3d[frame_i].to(x[i].device)
                if bboxes_3d.numel() == 0:
                    bboxes_3d = torch.zeros((0, 7), device=x[i].device)
            else:
                if self._log_missing_bboxes:
                    print(f"[WARN] missing gt bboxes_3d in train sample batch={i} frame={frame_i}")
                bboxes_3d = torch.zeros((0, 7), device=x[i].device)
            # Match bboxes_3d length to labels_3d for InstanceData consistency.
            if bboxes_3d.shape[0] + self.sem_len == n_labels:
                bboxes_3d = torch.cat([bboxes_3d, torch.zeros(self.sem_len, 7).to(x[i].device)])
            else:
                if bboxes_3d.shape[0] < n_labels:
                    pad = torch.zeros((n_labels - bboxes_3d.shape[0], 7), device=x[i].device)
                    bboxes_3d = torch.cat([bboxes_3d, pad], dim=0)
                elif bboxes_3d.shape[0] > n_labels:
                    bboxes_3d = bboxes_3d[:n_labels]
            temp.bboxes_3d = bboxes_3d
            gt_instances_.append(temp)

        if self.use_temporal_loss:
            with torch.no_grad():
                before_query_memory = [x[i].clone() for i in range(len(x))]
                before_mask_memory = [gt_instances_[i].sp_masks.clone() for i in range(len(gt_instances_))]
                before_sp_xyz = [sp_xyz[i].clone() for i in range(len(sp_xyz))]
            if self.before_mask_memory is not None:
                for i in range(len(x)):
                    gt_instances_[i].sp_masks = torch.cat([self.before_mask_memory[i], gt_instances_[i].sp_masks], dim=1) # [20000, 2]
                    sp_xyz[i] = torch.cat([self.before_sp_xyz[i], sp_xyz[i]], dim=0) # [20000, 3]
                    x[i] = torch.cat([self.before_query_memory[i], x[i]], dim=0) # [20000, 96]
            with torch.no_grad():
                self.before_query_memory = before_query_memory
                self.before_mask_memory = before_mask_memory
                self.before_sp_xyz = before_sp_xyz
        
        queries = []
        for i in range(len(x)): # batch_size
            if self.query_thr < 1: # 0.5 
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).ceil().int()
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                gt_instances_[i].query_masks = gt_instances_[i].sp_masks[:, ids]
                sp_xyz[i] = sp_xyz[i][ids]
            else:
                queries.append(x[i])
                gt_instances_[i].query_masks = gt_instances_[i].sp_masks
      
        return queries, gt_instances_, sp_xyz
    def _select_queries_predict(self, device, gt_instances, frame_i):
        """Select queries for train pass.
        """



        gt_instances_ = [] # 
        for i in range(len(gt_instances)): # batch_size
            temp = InstanceData()
            temp.labels_3d = gt_instances[i].labels_3d[frame_i].to(device)
            temp.sp_masks = gt_instances[i].sp_masks[frame_i].to(device)
            n_labels = int(temp.labels_3d.shape[0])
            if hasattr(gt_instances[i], "bboxes_3d"):
                bboxes_3d = gt_instances[i].bboxes_3d[frame_i].to(device)
                if bboxes_3d.numel() == 0:
                    bboxes_3d = torch.zeros((0, 7), device=device)
            else:
                if self._log_missing_bboxes:
                    print(f"[WARN] missing gt bboxes_3d in predict sample batch={i} frame={frame_i}")
                bboxes_3d = torch.zeros((0, 7), device=device)
            if bboxes_3d.shape[0] + self.sem_len == n_labels:
                bboxes_3d = torch.cat([bboxes_3d, torch.zeros(self.sem_len, 7).to(device)])
            else:
                if bboxes_3d.shape[0] < n_labels:
                    pad = torch.zeros((n_labels - bboxes_3d.shape[0], 7), device=device)
                    bboxes_3d = torch.cat([bboxes_3d, pad], dim=0)
                elif bboxes_3d.shape[0] > n_labels:
                    bboxes_3d = bboxes_3d[:n_labels]
            temp.bboxes_3d = bboxes_3d
            gt_instances_.append(temp)

        for i in range(len(gt_instances)): # batch_size
            gt_instances_[i].query_masks = gt_instances_[i].sp_masks
        return gt_instances_
    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def _process_frame_queries(self, track_instances, active_queries_num):
        """Process and update track_instances for the given frame."""
        # Handle query initialization and self-attention
        for i in range(len(track_instances.queries)):
            track_instances.queries[i] = self.input_proj(track_instances.queries[i])
            track_instances.queries[i] = self.self_attn(track_instances.queries[i],active_queries_num[i])
            track_instances.queries[i] = track_instances.queries[i].squeeze(0)
            track_instances.queries[i] = self.ffn(track_instances.queries[i])
            track_instances.queries[i] = self.output_proj(track_instances.queries[i])

        return track_instances

    def _init_query(self, queries, mot_type='motr'):

        track_instances = Instances((1, 1))
        device = next(self.backbone.parameters()).device

        # cls_preds must align with decoder output shape:
        # [num_queries, num_instance_classes + 1] (last is "no-object").
        # For CA (num_classes==1), this becomes 2, matching legacy behavior.
        cls_dim = int(getattr(self, "num_classes", 1)) + 1
        sem_dim = int(getattr(self, "sem_len", 201))

        fields = {
            'obj_idxes':        (None,    torch.long,    -1,    True),
            'matched_gt_idxes': (None,    torch.long,    -1,    True),
            'cls_preds':        (cls_dim, torch.float32, -1,    False),
            'sem_preds':        (sem_dim, torch.float32, -1,    False),
            'masks':            (20000,   torch.float32, -1,    False),
            'bboxes':           (6,       torch.float32, -1,    False),
            'fp_flag':          (None,    torch.bool,    False, True),
            'track_age':        (None,    torch.int,      0,    True),
        }

        for name, (feat_dim, dtype, fill_val, is_1d) in fields.items():
            data = [
                torch.full(
                    (q.shape[0],) if is_1d else (q.shape[0], feat_dim),
                    fill_val, dtype=dtype, device=device
                )
                for q in queries
            ]
            setattr(track_instances, name, data)

        if mot_type == 'dq_track':
            track_instances.queries = [
                torch.full((q.shape[0], 256), -1, dtype=torch.float32, device=device)
                for q in queries
            ]
        else:
            raise NotImplementedError(f"mot_type {mot_type} is not supported")

        return track_instances.to(device)
    def _init_query_test(self, queries, mot_type='motr', mode='train', fix_num=500, batch_size=1):
        track_instances = Instances((1, 1))
        device = next(self.backbone.parameters()).device
        cls_dim = int(getattr(self, "num_classes", 1)) + 1
        fields = {
            # 数值型字段: 填 -1
            'obj_idxes':        (None,    torch.long,    -1,    True),
            'obj_labels':       (None,    torch.long,    -1,    True), 
            'global_track_id': (None,    torch.long, -1,   True),
            'category':        (None,    torch.long,    -1,    True),
            # 'matched_gt_idxes': (None,    torch.long,    -1,    True),
            # 'current_obj_idxes': (None,    torch.long,    -1,    True),
            'cls_preds':        (cls_dim, torch.float32, -1,    False),
            'scores':           (None,    torch.float32, -1,    True),
            # 'sem_preds':        (201,     torch.float32, -1,    False),
            # 'masks':            (20000,   torch.float32, -1,    False),
            'bboxes':           (6,       torch.float32, -1,    False),
            # 布尔型或需要特殊初始值的
            'long_track':       (None,    torch.bool,    False, True),
            'valid_track':      (None,    torch.bool,    False, True),
            'active':           (None,    torch.bool,    False, True),
            # 浮点型零填充
            'track_age':        (None,    torch.int, 0,   True),
            'disappear_time':   (None,    torch.int, 0,   True),
            
        }


        for name, (feature_dim, dtype, fill_val, is_1d) in fields.items():
            data = [
                torch.full((fix_num,) if is_1d else (fix_num, feature_dim), fill_val, dtype=dtype, device=device)
                for batch_idx in range(batch_size)
            ]
            setattr(track_instances, name, data)

        # Store queries as a list
        if mot_type == 'dq_track':
            track_instances.queries = [torch.full((fix_num, 256), 0, dtype=torch.float32, device=device) for batch_idx in range(batch_size)]
        else:
            raise NotImplementedError(f"mot_type {mot_type} is not supported")

        return track_instances.to(device)
    def update_untracked_gt_instances(self, gt_instances, gt_point_instances, untracked_tgt_indexes, new_indexes):
        """Update and create untracked GT instances."""
        untracked_gt_instances, untracked_gt_points_instances = [], []

        untracked_tgt_indexes_gt = [
            torch.cat([untracked_tgt_indexes[i], new_indexes[i]], dim=0) 
            for i in range(len(untracked_tgt_indexes))
        ]
        for idx,gt in enumerate(gt_instances):
            new_gt = gt[untracked_tgt_indexes_gt[idx]]
            new_gt.labels_3d = gt.labels_3d[untracked_tgt_indexes_gt[idx]]
            new_gt.bboxes_3d = gt.bboxes_3d[untracked_tgt_indexes_gt[idx]]
            new_gt.sp_masks = gt.sp_masks[untracked_tgt_indexes_gt[idx]]
            new_gt.query_masks = gt.query_masks[untracked_tgt_indexes_gt[idx]]
            untracked_gt_instances.append(new_gt)

        for idx,gt_points in enumerate(gt_point_instances):
            new_gt_points = gt_points[untracked_tgt_indexes[idx]]
            new_gt_points.p_masks = gt_points.p_masks[untracked_tgt_indexes[idx]]
            untracked_gt_points_instances.append(new_gt_points)

        return untracked_gt_instances, untracked_gt_points_instances

    def _empty_single_track(self,active_track_instances):
        # track_instances = Instances((1,1))
        device = next(self.backbone.parameters()).device

        active_track_instances.obj_idxes.append(torch.empty(0, dtype=torch.long, device=device))
        active_track_instances.matched_gt_idxes.append(torch.empty(0, dtype=torch.long, device=device))
        # active_track_instances.scores.append(torch.empty(0, dtype=torch.float, device=device))
        active_track_instances.cls_preds.append(torch.empty(0, dtype=torch.float, device=device))
        active_track_instances.sem_preds.append(torch.empty(0, dtype=torch.float, device=device))
        active_track_instances.masks.append(torch.empty(0, dtype=torch.float, device=device))
        active_track_instances.bboxes.append(torch.empty(0, dtype=torch.float, device=device))
        active_track_instances.queries.append(torch.empty(0, dtype=torch.float, device=device))
        return active_track_instances.to(device)
    
    def _empty_query(self, track_instances_old):
        track_instances = Instances((1,1))
        device = next(self.backbone.parameters()).device
        for key_name in track_instances_old.get_fields().keys():
            setattr(track_instances, key_name, [])
        return track_instances.to(device)
    
    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        active_track_instances = self._empty_query(track_instances)
        for batch_idx in range(len(track_instances.matched_gt_idxes)):
            if self.training:
                active_idxes = (track_instances.matched_gt_idxes[batch_idx] >= 0)
                # active_track_instances = track_instances[active_idxes]
                for key_name in track_instances.get_fields().keys():
                    # track_instances[key_name] = track_instances[key_name][batch_idx][active_idxes]
                    active_value = getattr(track_instances, key_name)[batch_idx][active_idxes]
                    getattr(active_track_instances, key_name).append(active_value)


            else:
                active_track_instances = track_instances[track_instances.matched_gt_idxes[batch_idx] >= 0]

        return active_track_instances
    def update_track_instances(self, track_instances, gt_instances, indices, unmatched_track_idxes_list, untracked_tgt_indexes_list, gt_num_list, is_last, track_embedding_for_update_list=None, unmatched_track_embedding_for_update_list=None):
        track_aug=dict(
            # drop_prob=0,
            fp_ratio=0.5,
            # trans_noise=0.0,
            )
        for batch_idx in range(len(gt_instances)):             
            current_matched_pred_id = indices[batch_idx][0]
            current_matched_gt_id = indices[batch_idx][1] 
            before_tracked_num = (track_instances.matched_gt_idxes[batch_idx] >= 0).sum()
            new_matched_mask = ~torch.isin(current_matched_gt_id, track_instances.matched_gt_idxes[batch_idx][:before_tracked_num])
            new_matched_gt_index = new_matched_mask.nonzero(as_tuple=True)[0]
            use_aug = self.asso_config.get('use_aug', False)
            if len(new_matched_gt_index) == 0:
                if use_aug and track_aug['fp_ratio'] > 0:
                    fp_track = (track_instances.matched_gt_idxes[batch_idx] == 99999)
                    if fp_track.sum() > 0: 
                        track_instances.obj_idxes[batch_idx][fp_track] = -1
                        track_instances.matched_gt_idxes[batch_idx][fp_track] = -1
                continue
            
            new_matched_pred_index = current_matched_pred_id[new_matched_gt_index]
            
            track_instances.obj_idxes[batch_idx][new_matched_pred_index + before_tracked_num] = current_matched_gt_id[new_matched_gt_index]
            track_instances.matched_gt_idxes[batch_idx][new_matched_pred_index + before_tracked_num] = current_matched_gt_id[new_matched_gt_index]
            if track_embedding_for_update_list is not None:
                track_instances.queries[batch_idx][new_matched_pred_index + before_tracked_num] = track_embedding_for_update_list[batch_idx][new_matched_mask]
            # Data Augmentation
            if use_aug and track_aug['fp_ratio'] > 0:
                current_scores = F.softmax(track_instances.cls_preds[batch_idx], dim=-1)[:, 0]
                fp_track = (track_instances.matched_gt_idxes[batch_idx] == 99999)
                
                if fp_track.sum() > 0: 
                    track_instances.obj_idxes[batch_idx][fp_track] = -1
                    track_instances.matched_gt_idxes[batch_idx][fp_track] = -1
                # select background det embedding
                det_mask = torch.ones_like(current_scores).bool() #
                det_mask[:before_tracked_num] = False 
                det_mask[before_tracked_num + current_matched_pred_id] = False 
                # select fp embedding according to prob
                fp_mask = det_mask & (current_scores > 0.0) 
                current_scores[~det_mask] = 0 
                fp_num = int(fp_mask.sum() * track_aug['fp_ratio'])
                if fp_mask.sum() > fp_num: 
                    score_sort = torch.argsort(current_scores, descending=True)
                    assert score_sort[fp_num:].max() < len(fp_mask), f"score_sort[fp_num:]={score_sort[fp_num:]} len(fp_mask)={len(fp_mask)}"
                    fp_mask[score_sort[fp_num:]] = False 
                fp_indices = fp_mask.nonzero(as_tuple=True)[0] 
                if len(fp_indices) == 0:
                    continue
                track_instances.fp_flag[batch_idx][fp_indices] = True 
                
                track_instances.queries[batch_idx][fp_indices] = unmatched_track_embedding_for_update_list[batch_idx][fp_indices - before_tracked_num] #将假阳性目标的查询嵌入更新为 track_embedding_for_update_list 中对应的值
                track_instances.matched_gt_idxes[batch_idx][fp_indices] = 99999

        # select active tracks
        tmp = {} 
        tmp['track_instances'] = track_instances 
        if not is_last:
            out_track_instances = self._select_active_tracks(tmp) 
            # frame_res['track_instances'] = out_track_instances
        else:
            out_track_instances = None
            # frame_res['track_instances'] = None        
        # track_instances = frame_res['track_instances']
        return out_track_instances
    def update_track_instances_predict(self, track_instances, gt_instances, indices, is_last):

        for batch_idx in range(len(gt_instances)):             
            current_matched_pred_id = indices[batch_idx][0]
            current_matched_gt_id = indices[batch_idx][1] 
            before_tracked_num = (track_instances.matched_gt_idxes[batch_idx] >= 0).sum()
            new_matched_gt_index = (~torch.isin(current_matched_gt_id, track_instances.matched_gt_idxes[batch_idx][:before_tracked_num])).nonzero(as_tuple=True)[0]
            if len(new_matched_gt_index) == 0:
                continue
            new_matched_pred_index = current_matched_pred_id[new_matched_gt_index]
            track_instances.obj_idxes[batch_idx][new_matched_pred_index + before_tracked_num] = current_matched_gt_id[new_matched_gt_index]
            track_instances.matched_gt_idxes[batch_idx][new_matched_pred_index + before_tracked_num] = current_matched_gt_id[new_matched_gt_index]

        # 
        # select active tracks
        tmp = {} 
        tmp['track_instances'] = track_instances 
        if not is_last:
            out_track_instances = self._select_active_tracks(tmp) 
            # frame_res['track_instances'] = out_track_instances
        else:
            out_track_instances = None

        return out_track_instances
    def _check_param_diffs(self):
        curr_params = {
            name: p.detach().cpu().clone()
            for name, p in self.named_parameters()
        }
        curr_buffers = {
            name: b.detach().cpu().clone()
            for name, b in self.named_buffers()
            if "running_mean" in name or "running_var" in name
        }

        if self._prev_param_snapshot is None:
            self._prev_param_snapshot = curr_params
            self._prev_buffer_snapshot = curr_buffers
            return

        for name, prev in self._prev_param_snapshot.items():
            now = curr_params[name]
            diff = (now - prev).abs().view(-1)
            max_diff = diff.max().item()
            if max_diff != 0:
                print(f"[Param Δ]  {name:40s} max|Δ| = {max_diff:.5e}")

        for name, prev in self._prev_buffer_snapshot.items():
            now = curr_buffers[name]
            diff = (now - prev).abs().view(-1)
            max_diff = diff.max().item()
            if max_diff != 0:
                print(f"[Buffer Δ] {name:40s} max|Δ| = {max_diff:.5e}")

        self._prev_param_snapshot  = curr_params
        self._prev_buffer_snapshot = curr_buffers

    def get_loss_track(self, track_instances, current_dict, mot_type):
        loss_track = 0
        return loss_track
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.
        """
        if self.debug_mode:
            self._check_param_diffs()
        losses, merge_feat_n_frames, ins_masks_query_n_frames = {}, [], []
        num_frames = batch_inputs_dict['points'][0].shape[0]
        if hasattr(self, 'memory'):
            self.memory.reset()
        if self.use_query_memory:
            self.reset_query_memory()
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_mask_memory = None
            self.before_sp_xyz = None
            self.inst_dict = None
        if self.decoder.use_query_memory2:
            self.decoder.reset_query_memory2()
        if self.use_decouple:
            self.decoder.reset_decouple()
        if self.use_mot:
            use_after_features = True
            self.merge_type = 'count'
        if self.merge_sp_masks:
            merge_loss = torch.tensor(0.0).to(batch_inputs_dict['points'][0].device)
        for frame_i in range(num_frames): 
            if self.merge_sp_masks:
                current_sp_pts_mask = [bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples]
                current_pt_instace_mask = [bds.gt_pts_seg.pts_instance_mask[frame_i] for bds in batch_data_samples]
                merge_loss += self.merge_superpixels_train(batch_inputs_dict, batch_data_samples, frame_i, current_sp_pts_mask, current_pt_instace_mask)
                
                if not self.use_mot:
                    if frame_i != num_frames - 1:
                        continue
                    else:
                        loss = {'merge_mask_loss': merge_loss}
                        return loss
            else:
                merged_sp_pts_masks = None
                merged_sp_masks = None
            ## Backbone
            if self.use_mot:
                if self.mot_type == 'dq_track':
                    if self.asso_config.get('train_asso_only', True):
                        with frozen_inference(self.backbone), frozen_inference(self.memory):
                            x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
                    else:
                        raise NotImplementedError(f"mot_type {self.mot_type} and train_asso_only {self.asso_config.get('train_asso_only', True)} is not supported")

            else:
                x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
            # Optional: GDINO full forward for DACA-2D during training.
            gdino_daca2d_cfg = self._get_gdino_daca2d_cfg(is_train=True)
            query2d_feats = None
            query2d_pos = None
            try:
                if isinstance(gdino_daca2d_cfg, dict) and bool(gdino_daca2d_cfg.get("enable", False)):
                    q2d_feats, q2d_pos, stats = self._run_gdino_daca2d_for_frame(
                        batch_inputs_dict, batch_data_samples, int(frame_i), is_train=True
                    )
                    mode = str(gdino_daca2d_cfg.get("mode", "diag_only"))
                    if mode != "fuse" and not getattr(self, "_warned_gdino_daca2d_mode", False):
                        import warnings
                        warnings.warn(
                            f"[GDINO][daca2d] enable=True but mode={mode!r}. "
                            "No object-level injection will happen unless mode='fuse'."
                        )
                        self._warned_gdino_daca2d_mode = True
                    if mode != "fuse":
                        q2d_feats, q2d_pos = None, None
                    query2d_feats, query2d_pos = q2d_feats, q2d_pos
                    self._last_gdino_daca2d_stats = stats
            except Exception as e:
                self._last_gdino_daca2d_stats = {"frame": int(frame_i), "skipped": "exception", "error": repr(e)}
            ## GT-prepare
            gt_instances = [s.gt_instances_3d for s in batch_data_samples]

            gt_point_instances, ins_masks_query_batch = [], []
            for i in range(len(gt_instances)): # batch_size
                ins = batch_data_samples[i].gt_pts_seg.pts_instance_mask[frame_i] # [20000]
                if torch.sum(ins == -1) != 0: 
                    # Use global instance number for each frame
                    ins[ins == -1] = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                    ins = F.one_hot(ins)[:, :-1]
                else:
                    ins = F.one_hot(ins)
                    max_ids = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                    if ins.shape[1] < max_ids:
                        zero_pad = torch.zeros(ins.shape[0], max_ids - ins.shape[1]).to(ins.device)
                        ins = torch.cat([ins, zero_pad], dim=-1)
                ins = ins.bool().T # [3, 20000]
                gt_point = InstanceData()
                gt_point.p_masks = ins
                gt_point_instances.append(gt_point)
                # --- SegDINO3D-style: compute GT axis-aligned bbox (center+size) from GT masks + point coords ---
                # This provides runtime `bboxes_3d` even when the dataset does not store them.
                # Only instance rows are computed here (semantic rows are padded later in _select_queries).
                try:
                    xyz = batch_inputs_dict['points'][i][frame_i, :, :3]
                    if torch.is_tensor(xyz) and xyz.numel() > 0 and torch.is_tensor(ins) and ins.numel() > 0:
                        n_inst = int(ins.shape[0])
                        bboxes = xyz.new_zeros((n_inst, 7))
                        for k in range(n_inst):
                            m = ins[k]
                            if m.any():
                                pts = xyz[m]
                                mn = pts.min(dim=0).values
                                mx = pts.max(dim=0).values
                                center = (mn + mx) * 0.5
                                size = (mx - mn).clamp(min=0)
                                bboxes[k, :3] = center
                                bboxes[k, 3:6] = size
                        if not hasattr(gt_instances[i], 'bboxes_3d') or gt_instances[i].bboxes_3d is None:
                            gt_instances[i].bboxes_3d = [None for _ in range(num_frames)]
                        if isinstance(gt_instances[i].bboxes_3d, list):
                            gt_instances[i].bboxes_3d[frame_i] = bboxes
                except Exception:
                    pass
            ## Query
            if self.use_query_memory:
                x = self.query_memory_aggregation(x, sp_xyz)
            if self.use_self_attn:
                query_self = self.muti_scale_self_attn(sp_xyz, x, x, x, sp_xyz)
                x = [x[i] + query_self[i] for i in range(len(x))]
                x = [self.self_attn_relu(x[i]) for i in range(len(x))]

            queries, gt_instances, sp_xyz = self._select_queries(x, gt_instances, sp_xyz, frame_i)
            if self.use_mot:
                device = x[0].device
                is_last = frame_i == num_frames - 1 
                if frame_i == 0:
                    track_instances = self._init_query(queries, mot_type=self.mot_type)
                    active_queries_num =  [q.shape[0] for q in track_instances.queries]
                else: 
                    init_track_instances: Instances = track_instances
                    if len(init_track_instances.queries) > 0:
                        active_queries_num = [len(q) for q in init_track_instances.queries]
                    else: #active_queries = 0
                        active_queries_num = [len(q) for q in init_track_instances.queries]  # Set to 0 if queries is empty
                    track_instances = Instances.cat([init_track_instances, self._init_query(queries, mot_type=self.mot_type)])

            ## Decoder
            super_points = ([bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples], all_xyz_w) 
            if self.use_mot:
                if self.mot_type == 'dq_track' and self.asso_config.get('train_asso_only', True):
                    # Default stage2 trains association only (decoder frozen under no_grad).
                    # For finetuning new decoder-side modules (e.g. track-window STM),
                    # allow overriding this behavior via `asso_config.freeze_decoder=False`.
                    freeze_decoder = bool(self.asso_config.get('freeze_decoder', True))
                    ctx = frozen_inference(self.decoder) if freeze_decoder else nullcontext()
                    with ctx:
                        x = self.decoder(
                            x, point_features, queries, super_points,
                            use_temporal_loss=self.use_temporal_loss,
                            inst_dict=self.inst_dict if self.use_temporal_loss else None,
                            track_instances=track_instances,
                            query2d_feats=query2d_feats, query2d_pos=query2d_pos,
                            gdino_daca2d_cfg=gdino_daca2d_cfg,
                            sp_pos_list_override=getattr(self, "_last_sp_pos_wo_elastic_list", None),
                            query3d_pos=sp_xyz,
                        )
                else:
                    raise NotImplementedError(f"mot_type {self.mot_type} and train_asso_only {self.asso_config.get('train_asso_only', True)} is not supported")
            else:
                x = self.decoder(
                    x, point_features, queries, super_points,
                    use_temporal_loss=self.use_temporal_loss,
                    inst_dict=self.inst_dict if self.use_temporal_loss else None,
                    use_one2many=self.use_one2many,
                    track_instances=track_instances if self.use_mot else None,
                    query2d_feats=query2d_feats, query2d_pos=query2d_pos,
                    gdino_daca2d_cfg=gdino_daca2d_cfg,
                    sp_pos_list_override=getattr(self, "_last_sp_pos_wo_elastic_list", None),
                    query3d_pos=sp_xyz,
                ) # ! 还是这里？
            if self.use_mot:          
                untracked_tgt_indexes_list = []
                # new_indexes_list = []
                unmatched_track_idxes_list = []
                gt_num_list = []
                gt_obj_list = []
                for batch_idx in range(len(queries)):
                    if frame_i == 0:
                        abs_boxes = x['bboxes'][batch_idx].clone()
                        abs_boxes[:, :3] += sp_xyz[batch_idx][:, :3]
                        track_instances.bboxes[batch_idx][:, :] = abs_boxes
                        # track_instances.bboxes[batch_idx][:, :] = x['bboxes'][batch_idx]
                        track_instances.masks[batch_idx][:, :] = x['masks'][batch_idx]
                        track_instances.sem_preds[batch_idx][:, :] = x['sem_preds'][batch_idx]
                        track_instances.cls_preds[batch_idx][:, :] = x['cls_preds'][batch_idx]
                        if self.mot_type == 'dq_track':
                            if not use_after_features:
                                track_instances.queries[batch_idx][:, :] = x['queries'][batch_idx]
                        else:
                            raise NotImplementedError(f"mot_type {self.mot_type} is not supported")
                    else:
                        abs_boxes = x['bboxes'][batch_idx].clone()
                        abs_boxes[:, :3] += sp_xyz[batch_idx][:, :3] 
                        track_instances.bboxes[batch_idx][active_queries_num[batch_idx]:, :] = abs_boxes
                        # track_instances.bboxes[batch_idx][active_queries_num[batch_idx]:, :] = x['bboxes'][batch_idx]
                        track_instances.masks[batch_idx][active_queries_num[batch_idx]:, :] = x['masks'][batch_idx]
                        track_instances.sem_preds[batch_idx][active_queries_num[batch_idx]:, :] = x['sem_preds'][batch_idx]
                        track_instances.cls_preds[batch_idx][active_queries_num[batch_idx]:, :] = x['cls_preds'][batch_idx]
                        if self.mot_type == 'dq_track':
                            if not use_after_features:
                                track_instances.queries[batch_idx][active_queries_num[batch_idx]:, :] = x['queries'][batch_idx]
                        else:
                            raise NotImplementedError(f"mot_type {self.mot_type} is not supported")

                if self.mot_type == 'dq_track':
                    track_embedding_for_update_list = []
                    if self.asso_config.get('use_aug', False):
                        unmatched_track_embedding_for_update_list = []
                    matched_list = []
                    for batch_idx in range(len(gt_instances)):
                        # valid_track = track_instances.matched_gt_idxes[batch_idx] >= 0
                        matched_list.append({
                            'track_idx': torch.empty(0, dtype=torch.int64, device=device),
                            'current_obj_idxes': torch.empty(0, dtype=torch.int64, device=device),
                            'gt_idx': torch.empty(0, dtype=torch.int64, device=device),
                            'valid_gt_idx':(gt_instances[batch_idx].labels_3d[:-201] != -1).nonzero(as_tuple=True)[0],
                            'mot_type': self.mot_type,})
                    indices = match_for_indices(self.matcher, x, gt_instances, gt_point_instances, matched_list)

                    loss_asso = torch.tensor(0.0).to(device)
                    if self.use_relative_asso:
                        loss_rel_asso = torch.tensor(0.0).to(device)
                    
                    for batch_idx in range(len(indices)):
                        pred_indices = indices[batch_idx][0]
                        if self.asso_config.get('use_aug', False):
                            all_pred_indices = torch.arange(len(x['queries'][batch_idx])).to(device)
                            unmatched_pred_indices = all_pred_indices
                            unmatched_det_embedding  = x['queries'][batch_idx][unmatched_pred_indices].clone()
                            unmatched_track_embedding_for_update = self.tracklet_trans(unmatched_det_embedding)
                            unmatched_obj_embedding = self.detector_trans(unmatched_det_embedding)
                            unmatched_det_bboxes = x['bboxes'][batch_idx][unmatched_pred_indices]
                            unmatched_det_bboxes[:, :3] += sp_xyz[batch_idx][unmatched_pred_indices][:, :3] 
                            unmatched_pos_embedding = self.box_trans(unmatched_det_bboxes)
                            unmatched_track_embedding_for_update, _ = self.query_inter(unmatched_track_embedding_for_update, unmatched_obj_embedding, unmatched_pos_embedding, unmatched_det_bboxes[:, :3])
                            unmatched_track_embedding_for_update_list.append(unmatched_track_embedding_for_update)


                        gt_indices = indices[batch_idx][1]
                        det_embedding = x['queries'][batch_idx][pred_indices].clone()
                        track_embedding_for_update = self.tracklet_trans(det_embedding)
                        obj_embedding = self.detector_trans(det_embedding)
                        det_bboxes = x['bboxes'][batch_idx][pred_indices]
                        det_bboxes[:, :3] += sp_xyz[batch_idx][pred_indices][:, :3] 
                        pos_embedding = self.box_trans(det_bboxes)
                        track_embedding_for_update, obj_embedding = self.query_inter(track_embedding_for_update, obj_embedding, pos_embedding, det_bboxes[:, :3])
                        valid_track = track_instances.matched_gt_idxes[batch_idx] >= 0
                        if self.use_relative_asso and valid_track.sum() > 0 and len(x['queries'][batch_idx]) > 0:
                            all_det_embedding = x['queries'][batch_idx].clone()
                            all_track_embedding_for_update = self.tracklet_trans(all_det_embedding)
                            all_obj_embedding = self.detector_trans(all_det_embedding)
                            all_det_bboxes = x['bboxes'][batch_idx].clone()
                            all_det_bboxes[:, :3] += sp_xyz[batch_idx][:, :3] 
                            all_pos_embedding = self.box_trans(all_det_bboxes)
                            all_track_embedding_for_update, all_obj_embedding = self.query_inter(all_track_embedding_for_update, all_obj_embedding, all_pos_embedding, all_det_bboxes[:, :3])
                            
                            valid_track = track_instances.matched_gt_idxes[batch_idx] >= 0
                            all_track_pos = track_instances.bboxes[batch_idx][valid_track][:, :3]
                            all_track_embedding = track_instances.queries[batch_idx][valid_track]
                            if self.use_bbox:
                                all_rel_dist = self.iou_calculator(bbox_pred_to_bbox(all_det_bboxes), bbox_pred_to_bbox(track_instances.bboxes[batch_idx][valid_track]))
                                all_rel_dist = all_rel_dist.unsqueeze(-1)
                            else:
                                all_det_pos = all_det_bboxes[:, :3]
                                all_rel_dist = (all_det_pos[:,None] - all_track_pos[None])**2
                                all_rel_dist = all_rel_dist.sum(-1, keepdim=True).sqrt()
                            all_geometry_embedding = self.rel_dist_embed(all_rel_dist)
                            all_appear_embedding = all_obj_embedding[:,None] * all_track_embedding[None]
                            all_fused_embedding = all_appear_embedding + all_geometry_embedding
                            all_det2track_heatmap = self.embed_trans2(all_fused_embedding).sum(-1) # [N_det, N_track]
                            target_query_masks = gt_instances[batch_idx].query_masks[:-self.sem_len, :].clone().T.float() # [gt_num, N_det]
                            nonzero_per_row = [
                                torch.nonzero(target_query_masks[i], as_tuple=True)[0].tolist()
                                for i in range(target_query_masks.size(0))
                            ]
                            track_inds = track_instances.matched_gt_idxes[batch_idx][valid_track]
                            target_heatmap = torch.zeros_like(all_det2track_heatmap) # [N_det, N_track]
                            for i in range(target_query_masks.size(0)):
                                cols = torch.nonzero(target_query_masks[i], as_tuple=True)[0]  # e.g. Tensor([2,5,7], device=...)
                                mask_tracks = torch.isin(track_inds, cols)                   # Bool Tensor, shape [num_tracks]
                                track_index = torch.nonzero(mask_tracks, as_tuple=True)[0] 
                                target_heatmap[i, track_index] = 1.0
                            rel_w = float(self.asso_config.get('rel_asso_loss_weight', 1.0))
                            loss_rel_asso += rel_w * self.heatmap_loss_fn(all_det2track_heatmap, target_heatmap)

                        
                        if valid_track.sum() > 0:
                            track_pos = track_instances.bboxes[batch_idx][valid_track][:, :3]
                            track_embedding = track_instances.queries[batch_idx][valid_track]
                            
                            if self.use_bbox:
                                rel_dist = self.iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(track_instances.bboxes[batch_idx][valid_track]))
                                rel_dist = rel_dist.unsqueeze(-1) # [N_det, N_track, 1]
                            else:
                                det_pos = det_bboxes[:, :3] # TODO 查看是否需要偏移,xyz有没有加进来
                                rel_dist = (det_pos[:,None] - track_pos[None])**2  # [N_det, N_track, 3]
                                rel_dist = rel_dist.sum(-1, keepdim=True).sqrt() # [N_det, N_track, 1]
                            geometry_embedding = self.rel_dist_embed(rel_dist) # [N_det, N_track, 1] -> [N_det, N_track, 256]
                            appear_embedding = obj_embedding[:,None] * track_embedding[None] # [N_det, N_track, 256]
                            fused_embedding = appear_embedding + geometry_embedding
                            det2track_mat = self.embed_trans(fused_embedding).sum(-1)

                            
                        else:
                            det2track_mat = None
                        track_embedding_for_update_list.append(track_embedding_for_update)
                        if det2track_mat is not None and valid_track.sum() > 0:
                            gt_per_frame = torch.full((len(det2track_mat),), -1).to(det2track_mat.device)
                            track_inds = track_instances.matched_gt_idxes[batch_idx][valid_track]
                            for _idx, obj_id in enumerate(gt_indices):
                                if obj_id not in track_inds: # 在一维张量 track_inds 中寻找等于 obj_id 的位置，并将该位置索引赋值给 gt_per_frame[_idx]。
                                    continue
                                gt_per_frame[_idx] = (track_inds==obj_id).nonzero(as_tuple=True)[0][0]
                            
                            # filter out new-born object
                            gt_mask = (gt_per_frame >= 0)
                            det2track_mat = det2track_mat[gt_mask] # [det_num, track_num]
                            gt_per_frame = gt_per_frame[gt_mask] # [det_num]
                            # assert gt_per_frame.max() < det2track_mat.shape[1]
                            if len(det2track_mat) > 0:
                                loss_asso += self.asso_config.get('asso_loss_weight', 0.5) * self.loss_asso(det2track_mat, gt_per_frame.long())
                            else:
                                loss_asso += 0

                            if use_after_features:
                                update_type = self.asso_config.get('update_type', 'ema')
                                update_rate = 1 - self.asso_config.get('no_update_rate', 0.0)
                                mask_upd = torch.rand(gt_per_frame.size(0), device=gt_per_frame.device) < update_rate
                                update_index = gt_per_frame[mask_upd]
                                if update_type == 'ema':
                                    track_instances.queries[batch_idx][update_index] = track_instances.queries[batch_idx][update_index] * self.ema_decay_rate + \
                                        (1 - self.ema_decay_rate) * track_embedding_for_update[gt_mask][mask_upd]
                                    track_instances.bboxes[batch_idx][update_index] = track_instances.bboxes[batch_idx][update_index] * self.ema_decay_rate + \
                                        (1 - self.ema_decay_rate) * det_bboxes[gt_mask][mask_upd]
                                elif update_type == 'count':
                                    track_instances.track_age[batch_idx][update_index] += 1
                                    track_age = track_instances.track_age[batch_idx][update_index].unsqueeze(1)
                                    track_instances.queries[batch_idx][update_index] = (track_instances.queries[batch_idx][update_index] * track_age + track_embedding_for_update[gt_mask][mask_upd]) / (track_age + 1)
                                    track_instances.bboxes[batch_idx][update_index] = (track_instances.bboxes[batch_idx][update_index] * track_age + det_bboxes[gt_mask][mask_upd]) / (track_age + 1)
                                else:
                                    raise NotImplementedError(f"Unknown update_type: {self.update_type}")
                        else:
                            pass
                else:
                    raise NotImplementedError(f"Unknown mot_type: {self.mot_type}")

                # track_loss = self.get_loss_track(track_instances, current_dict, mot_type=self.mot_type)
                # update the track_instances
                track_instances = self.update_track_instances(track_instances, gt_instances, indices, unmatched_track_idxes_list, untracked_tgt_indexes_list, gt_num_list, is_last,
                                                               track_embedding_for_update_list if (self.mot_type == 'dq_track' and use_after_features) else None,
                                                               unmatched_track_embedding_for_update_list if self.asso_config.get('use_aug', False) else None)
                

            ## Query projector
            for i in range(len(gt_instances)):
                ins_masks_query = gt_instances[i].query_masks[:-self.sem_len, :]
                ins_masks_query = [ins_masks_query[i].nonzero().flatten()
                        for i in range(ins_masks_query.shape[0])]
                ins_masks_query_batch.append(ins_masks_query)
            if hasattr(self, 'merge_head'): # True
                merge_feat = self.merge_head(x['queries'])
                merge_feat_n_frames.append(merge_feat)
                ins_masks_query_n_frames.append(ins_masks_query_batch)
            
            ## Loss
            if self.use_temporal_loss:
                loss, inst_dict = self.criterion(x, gt_instances, gt_point_instances, sp_xyz, self.decoder.mask_pred_mode, use_temporal_loss=self.use_temporal_loss)
                inst_dict['inst_pred_masks'] = []
                inst_dict['inst_points'] = []
                inst_dict['correspond_inst_gt'] = []
                inst_dict['bboxes_3d'] = []
                inst_dict['querys'] = []
                for i in range(len(gt_instances)):
                    # 选出保留的id(socre + matched)
                    pred_scores = x['cls_preds'][i].sigmoid()[:, :-1]
                    scores, topk_idx = pred_scores.flatten(0, 1).topk(min(self.test_cfg.topk_insts, pred_scores.shape[0]), sorted=False)
                    matched_idx = inst_dict['indices'][i][0]
                    mask = ~torch.isin(topk_idx, matched_idx)
                    filtered_idx = topk_idx[mask]
                    correspond_inst_gt = torch.cat([inst_dict['indices'][i][1], -1 * torch.ones(filtered_idx.shape[0], device=filtered_idx.device)], dim=0)
                    all_idx = torch.cat([matched_idx, filtered_idx], dim=0)
                    scores = torch.cat([torch.ones(matched_idx.shape[0], device=scores.device), scores[mask]], dim=0)
                    labels = torch.arange(self.num_classes, device=scores.device).unsqueeze(0).repeat(len(all_idx), 1).flatten(0, 1)
                    

                    # nms 进行过滤
                    inst_pred_mask = x['masks'][i].sigmoid() > self.test_cfg.sp_score_thr
                    inst_pred_mask = inst_pred_mask[all_idx]
                    kernel = self.test_cfg.matrix_nms_kernel
                    scores, labels, inst_pred_mask, keep_inds = mask_matrix_nms(
                        inst_pred_mask, labels, scores, kernel=kernel)
                    all_idx = all_idx[keep_inds]
                    correspond_inst_gt = correspond_inst_gt[keep_inds]

                    # 利用点的数量进行过滤
                    mask_pointnum = inst_pred_mask.sum(1) > self.test_cfg.npoint_thr
                    # scores = scores[mask_pointnum]
                    # labels = labels[mask_pointnum]
                    inst_pred_mask = inst_pred_mask[mask_pointnum]
                    all_idx = all_idx[mask_pointnum]
                    correspond_inst_gt = correspond_inst_gt[mask_pointnum]

                    bboxes_3d = x['bboxes'][i][all_idx].detach()
                    bboxes_3d[:, :3] += x['centers'][i][all_idx].detach()
                    inst_dict['bboxes_3d'].append(bboxes_3d)
                    inst_dict['inst_pred_masks'].append(inst_pred_mask.detach())
                    inst_dict['correspond_inst_gt'].append(correspond_inst_gt.detach())
                    inst_dict['querys'].append(x['queries'][i][all_idx].detach()) # [N, 96]

                    # 每个物体对应的点

                    inst_dict['inst_points'].append([])
                    current_points = batch_inputs_dict['points'][i][frame_i]
                    for obj_id in range(inst_pred_mask.shape[0]):
                        inst_dict['inst_points'][i].append(current_points[inst_pred_mask[obj_id]])
                self.inst_dict = inst_dict
                

            else:
                if self.use_mot:
                    if self.mot_type == 'dq_track':
                        if self.asso_config.get('train_asso_only', True):
                            # loss = loss_asso
                            loss = {'loss_asso': loss_asso}
                            if self.use_relative_asso:
                                loss.update({'loss_rel_asso': loss_rel_asso})
                        else:
                            loss = self.criterion(x, gt_instances, gt_point_instances, sp_xyz, self.decoder.mask_pred_mode) # + loss_asso
                            loss.update({'loss_asso': loss_asso})
                    else:
                        raise NotImplementedError(f"Unknown mot_type: {self.mot_type}")
                else:
                    loss = self.criterion(x, gt_instances, gt_point_instances, sp_xyz, self.decoder.mask_pred_mode)
                    if self.use_one2many:
                        loss_one2many = self.criterion_one2many(x['one2many_outputs'], gt_instances, gt_point_instances, sp_xyz, self.decoder.mask_pred_mode, use_one2many=self.use_one2many)
                        for key, value in loss_one2many.items():
                            loss_one2many[key] = value * self.one2many_loss_weight
                        loss.update(loss_one2many)
                    if self.reweight_dict is not None:
                        for key in loss.keys():
                            if key in self.reweight_dict:
                                loss[key] = loss[key] * self.reweight_dict[key]
            # print(loss)
            for key, value in loss.items():
                if key in losses:
                    losses[key] += value
                else:
                    losses[key] = value
        ## Query contrast 计算同一个物体的对比损失
        if hasattr(self, 'merge_criterion'): # True
            merge_feat_n_frames = [[frame[i] for frame in merge_feat_n_frames]
                 for i in range(len(merge_feat_n_frames[0]))]
            ins_masks_query_n_frames = [[frame[i] for frame in ins_masks_query_n_frames]
                 for i in range(len(ins_masks_query_n_frames[0]))]
            loss = self.merge_criterion(merge_feat_n_frames, ins_masks_query_n_frames)
            losses.update(loss)
        return losses
    def query_memory_aggregation(self, x, sp_xyz):
        if self.query_memory is not None and self.pos_memory is not None:
            query_x = self.muti_scale_query(sp_xyz, x, self.query_memory, self.query_memory, self.pos_memory)
            x = [x[i] + query_x[i] for i in range(len(x))]
            x = [self.query_memory_relu(x[i]) for i in range(len(x))]
        detach_query = [x[i].clone() for i in range(len(x))]
        detach_pos = [sp_xyz[i].clone() for i in range(len(sp_xyz))]
        self.query_memory = detach_query
        self.pos_memory = detach_pos
        return x
    def query_memory_aggregation_predict(self, x, sp_xyz):
        if self.query_memory is not None and self.pos_memory is not None:
            query_x = self.muti_scale_query(sp_xyz, x, self.query_memory, self.query_memory, self.pos_memory)
            x = [x[i] + query_x[i] for i in range(len(x))]
            x = [self.query_memory_relu(x[i]) for i in range(len(x))]
        with torch.no_grad():
            detach_query = [x[i].clone().detach() for i in range(len(x))]
            detach_pos = [sp_xyz[i].clone().detach() for i in range(len(sp_xyz))]
            self.query_memory = detach_query
            self.pos_memory = detach_pos
        return x
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        assert len(batch_data_samples) == 1
        results, query_feats_list, sem_preds_list, sp_xyz_list, bboxes_list, cls_preds_list = [], [], [], [], [], []
        num_frames = batch_inputs_dict['points'][0].shape[0]
        if hasattr(self, 'memory'):
            self.memory.reset()
        if self.use_query_memory:
            self.reset_query_memory()
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_mask_memory = None
            self.before_sp_xyz = None
        if self.use_temporal_loss:
            self.inst_dict = None
        if self.use_mot:
            self.current_max_track_id = 0
        # Optional: per-scene monitoring (side-channel; does not affect outputs).
        online_monitor_cfg = self.test_cfg.get('online_monitor', None) or {}
        online_monitor_enable = bool(online_monitor_cfg.get('enable', False))
        online_monitor = None
        if online_monitor_enable:
            meta = getattr(batch_data_samples[0], 'img_metas', None)
            if not isinstance(meta, dict):
                try:
                    meta = batch_data_samples[0].metainfo
                except Exception:
                    meta = {}
            scene_id = (
                meta.get('scene_id', None)
                or meta.get('scan_id', None)
                or meta.get('sample_idx', None)
                or meta.get('lidar_idx', None)
                or meta.get('ann_file', None)
                or meta.get('pts_filename', None)
                or 'unknown'
            )
            online_monitor = {
                "scene_id": str(scene_id),
                "num_frames": int(num_frames),
                "test_cfg": {
                    "merge_sp_masks": bool(getattr(self, 'merge_sp_masks', False)),
                    "use_bbox": bool(getattr(self, 'use_bbox', False)),
                    "merge_type": str(self.test_cfg.get('merge_type', '')),
                    "topk_insts": int(self.test_cfg.get('topk_insts', -1)),
                    "inst_score_thr": float(self.test_cfg.get('inst_score_thr', 0.0)),
                    "pan_score_thr": float(self.test_cfg.get('pan_score_thr', 0.0)),
                    "sp_score_thr": float(self.test_cfg.get('sp_score_thr', 0.0)),
                    "npoint_thr": int(self.test_cfg.get('npoint_thr', 0)),
                    "obj_normalization": bool(self.test_cfg.get('obj_normalization', False)),
                    "inscat_topk_insts": int(self.test_cfg.get('inscat_topk_insts', -1)),
                    "nms": bool(self.test_cfg.get('nms', False)),
                    "matrix_nms_kernel": str(self.test_cfg.get('matrix_nms_kernel', '')),
                },
                "monitor_cfg": {
                    k: v
                    for k, v in (dict(online_monitor_cfg).items() if isinstance(online_monitor_cfg, dict) else [])
                    if isinstance(v, (bool, int, float, str))
                },
                "frames": [],
            }
        # Reset per-scene GDINO diagnostics counter (side-channel only).
        try:
            self._gdino_diag_seen = 0
        except Exception:
            pass
        # Reset per-scene GDINO DACA-2D counter (side-channel only).
        try:
            self._gdino_daca2d_seen = 0
        except Exception:
            pass
        # Optional: GT-aligned embedding stability diagnostics (side-channel only).
        gt_emb_diag_cfg = {}
        gt_emb_diag_state = None
        gt_emb_diag_by_frame = {}
        if online_monitor_enable and isinstance(online_monitor_cfg, dict):
            gt_emb_diag_cfg = online_monitor_cfg.get("gt_emb_diag", {}) or {}
            if isinstance(gt_emb_diag_cfg, dict) and bool(gt_emb_diag_cfg.get("enable", False)):
                gt_emb_diag_state = {"prev": {}}
        for frame_i in range(num_frames):
            ## Backbone + SP merge (optional)  -> features, point_features, all_xyz_w, sp_xyz
            if self.merge_sp_masks:
                # Integrated path: merge SPs and extract features once
                x, point_features, all_xyz_w, sp_xyz = self.merge_superpixels_extract_feat(
                    batch_inputs_dict, batch_data_samples, frame_i)
            else:
                x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
            # Optional: GroundingDINO projection diagnostics (side-channel only).
            try:
                self._last_gdino_diag_stats = self._run_gdino_diag_for_frame(
                    batch_inputs_dict, batch_data_samples, int(frame_i)
                )
            except Exception as e:
                self._last_gdino_diag_stats = {"frame": int(frame_i), "skipped": "exception", "error": repr(e)}
            # Optional: GDINO full forward for DACA-2D (side-channel by default).
            gdino_daca2d_cfg = {}
            try:
                gdino_daca2d_cfg = (self.test_cfg or {}).get("gdino_daca2d", {}) or {}
            except Exception:
                gdino_daca2d_cfg = {}
            query2d_feats = None
            query2d_pos = None
            try:
                if isinstance(gdino_daca2d_cfg, dict) and bool(gdino_daca2d_cfg.get("enable", False)):
                    q2d_feats, q2d_pos, stats = self._run_gdino_daca2d_for_frame(
                        batch_inputs_dict, batch_data_samples, int(frame_i), is_train=False
                    )
                    mode = str(gdino_daca2d_cfg.get("mode", "diag_only"))
                    if mode != "fuse":
                        q2d_feats, q2d_pos = None, None
                    query2d_feats, query2d_pos = q2d_feats, q2d_pos
                    self._last_gdino_daca2d_stats = stats
                else:
                    self._last_gdino_daca2d_stats = None
            except Exception as e:
                self._last_gdino_daca2d_stats = {"frame": int(frame_i), "skipped": "exception", "error": repr(e)}
            ## Query
            if self.use_query_memory:
                x = self.query_memory_aggregation_predict(x, sp_xyz)
            if self.use_self_attn:
                query_self = self.muti_scale_self_attn(sp_xyz, x, x, x, sp_xyz)
                x = [x[i] + query_self[i] for i in range(len(x))]
                x = [self.self_attn_relu(x[i]) for i in range(len(x))]
            if self.use_mot:
                device = x[0].device
                is_last = frame_i == num_frames - 1 

                if frame_i == 0:
                    track_instances = self._init_query_test(x, mot_type=self.mot_type, mode='test')
                    active_queries_num =  [q.shape[0] for q in track_instances.queries]
                else:
                    init_track_instances: Instances = track_instances
                    if len(init_track_instances.queries) > 0:
                        active_queries_num = [len(q) for q in init_track_instances.queries]
                    else: #active_queries = 0
                        active_queries_num = [len(q) for q in init_track_instances.queries]  # Set to 0 if queries is empty
            ## Decoder 
            super_points = ([bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples], all_xyz_w) # ([20000], [20000, 1])
            # Optional: enable decoder-side diagnostics for track-window STM.
            try:
                trk_stm_mon = online_monitor_cfg.get("trk_stm", {}) if isinstance(online_monitor_cfg, dict) else {}
                self.decoder._trk_stm_diag_collect = bool(trk_stm_mon.get("enable", False)) and online_monitor_enable
            except Exception:
                pass
            x = self.decoder(
                x, point_features, x, super_points,
                track_instances=track_instances if self.use_mot else None,
                query2d_feats=query2d_feats, query2d_pos=query2d_pos,
                gdino_daca2d_cfg=gdino_daca2d_cfg,
                sp_pos_list_override=getattr(self, "_last_sp_pos_wo_elastic_list", None),
                query3d_pos=sp_xyz,
            ) # [N_segment, 96] [20000, 99] [N_segment, 96] ([20000], [20000, 1])
            ## Post-processing
            if online_monitor_enable:
                # Provide per-frame GT instance ids to post-processing for optional GT-aware stats.
                # This is strictly for monitoring and must not affect prediction results.
                try:
                    gt_inst = batch_data_samples[0].gt_pts_seg.pts_instance_mask[frame_i]
                except Exception:
                    gt_inst = None
                self._monitor_frame_ctx = {
                    "frame": int(frame_i),
                    "gt_inst": gt_inst,
                    "gt_vis_npoint": int(online_monitor_cfg.get("gt_vis_npoint", 100)),
                    "iou_thr": float(online_monitor_cfg.get("iou_thr", 0.5)),
                    "iou_lo_thr": float(online_monitor_cfg.get("iou_lo_thr", 0.1)),
                    "gt_frame_stride": int(online_monitor_cfg.get("gt_frame_stride", 5)),
                }
            pred_pts_seg, mapping = self.predict_by_feat(
                x, batch_data_samples[0].gt_pts_seg.sp_pts_mask[frame_i])
            results.append(pred_pts_seg[0])
            # Optional: GT-aligned embedding stability diagnostics (oracle association by GT IoU).
            if gt_emb_diag_state is not None and isinstance(gt_emb_diag_cfg, dict):
                try:
                    pm = pred_pts_seg[0].get("pts_instance_mask", None) if hasattr(pred_pts_seg[0], "get") else None
                    pq = pred_pts_seg[0].get("instance_queries", None) if hasattr(pred_pts_seg[0], "get") else None
                    if isinstance(pm, (list, tuple)) and len(pm) > 0 and isinstance(pq, (list, tuple)) and len(pq) > 0:
                        _cfg = dict(gt_emb_diag_cfg)
                        _cfg.setdefault("gt_vis_npoint", int(online_monitor_cfg.get("gt_vis_npoint", 100)))
                        _cfg.setdefault("iou_thr", float(online_monitor_cfg.get("iou_thr", 0.5)))
                        _cfg.setdefault("iou_lo_thr", float(online_monitor_cfg.get("iou_lo_thr", 0.1)))
                        _cfg.setdefault("frame_stride", int(online_monitor_cfg.get("gt_frame_stride", 1)))
                        gt_emb_diag_by_frame[int(frame_i)] = self._run_gt_emb_diag_for_frame(
                            gt_inst=gt_inst,
                            pred_masks=pm[0],
                            pred_queries=pq[0],
                            frame_i=int(frame_i),
                            state=gt_emb_diag_state,
                            cfg=_cfg,
                        )
                    else:
                        gt_emb_diag_by_frame[int(frame_i)] = {"frame": int(frame_i), "skipped": "no_pred_fields"}
                except Exception as e:
                    gt_emb_diag_by_frame[int(frame_i)] = {"frame": int(frame_i), "skipped": "exception", "err": repr(e)}
            if self.use_mot:  
                batch_idx = 0
                if self.asso_config.get('debug', False):
                    gt_instances = [s.gt_instances_3d for s in batch_data_samples] 
                    gt_point_instances, matched_list = [], []
                    for i in range(len(gt_instances)): # batch_size
                        ins = batch_data_samples[i].gt_pts_seg.pts_instance_mask[frame_i] # [20000]
                        if torch.sum(ins == -1) != 0: 
                            # Use global instance number for each frame
                            ins[ins == -1] = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                            ins = F.one_hot(ins)[:, :-1]
                        else:
                            ins = F.one_hot(ins)
                            max_ids = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                            if ins.shape[1] < max_ids:
                                zero_pad = torch.zeros(ins.shape[0], max_ids - ins.shape[1]).to(ins.device)
                                ins = torch.cat([ins, zero_pad], dim=-1)
                        ins = ins.bool().T # [3, 20000]
                        gt_point = InstanceData()
                        gt_point.p_masks = ins
                        gt_point_instances.append(gt_point)
                    gt_instances = self._select_queries_predict(device, gt_instances, frame_i) 
                    matched_list.append({
                        'track_idx': torch.empty(0, dtype=torch.int64, device=device),
                        'current_obj_idxes': torch.empty(0, dtype=torch.int64, device=device),
                        'gt_idx': torch.empty(0, dtype=torch.int64, device=device),
                        'valid_gt_idx':(gt_instances[batch_idx].labels_3d[:-201] != -1).nonzero(as_tuple=True)[0],
                        'mot_type': self.mot_type,})
                    indices = match_for_indices(self.matcher, x, gt_instances, gt_point_instances, matched_list)  
                if self.mot_type == 'dq_track':

                    valid_track = track_instances.valid_track[0].clone()
                    valid_det = mapping[0]
                    
                    det_embedding = x['queries'][batch_idx][valid_det]
                    det_category = x['sem_preds'][batch_idx][valid_det].argmax(1)
                    track_embedding_for_update = self.tracklet_trans(det_embedding)
                    obj_embedding = self.detector_trans(det_embedding)
                    det_bboxes = x['bboxes'][batch_idx][valid_det]
                    det_bboxes[:, :3] += sp_xyz[batch_idx][valid_det]
                    pos_embedding = self.box_trans(det_bboxes)
                    track_embedding_for_update, obj_embedding = self.query_inter(track_embedding_for_update, obj_embedding, pos_embedding, det_bboxes[:, :3])

                    if valid_track.sum() > 0:
                        track_pos = track_instances.bboxes[batch_idx][valid_track][:, :3]
                        track_embedding = track_instances.queries[batch_idx][valid_track]
                        track_category = track_instances.category[batch_idx][valid_track]
                        # det_pos = det_bboxes[:, :3] # TODO 查看是否需要偏移
                        # rel_dist = (det_pos[:,None] - track_pos[None])**2  # [N_det, N_track, 3]
                        # rel_dist = rel_dist.sum(-1, keepdim=True).sqrt() # [N_det, N_track, 1]
                        if self.use_bbox:
                            rel_dist = self.iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(track_instances.bboxes[batch_idx][valid_track]))
                            rel_dist = rel_dist.unsqueeze(-1) # [N_det, N_track, 1]
                        else:
                            det_pos = det_bboxes[:, :3] # TODO 查看是否需要偏移
                            rel_dist = (det_pos[:,None] - track_pos[None])**2  # [N_det, N_track, 3]
                            rel_dist = rel_dist.sum(-1, keepdim=True).sqrt() # [N_det, N_track, 1]
                        geometry_embedding = self.rel_dist_embed(rel_dist) # [N_det, N_track, 1] -> [N_det, N_track, 256]
                        appear_embedding = obj_embedding[:,None] * track_embedding[None] # [N_det, N_track, 256]
                        fused_embedding = appear_embedding + geometry_embedding
                        det2track_mat = self.embed_trans(fused_embedding).sum(-1)
                        det2track_mat = det2track_mat.softmax(1)
                        if self.use_relative_asso:
                            det2track_heatmap = self.embed_trans2(fused_embedding).sum(-1).sigmoid() # [N_det, N_track]
                            det2track_mat = det2track_mat * det2track_heatmap

                        # Scheme A: compute det2buffer_mat with buffer snapshots (align with det2track path)
                        det2buffer_mat = None
                        if 'online_merger' in locals() and getattr(online_merger, 'use_buffer', False):
                            buf_ids, buf_queries, buf_bboxes, buf_cats = online_merger.get_buffer_snapshots(device=det_embedding.device)
                            if buf_queries is not None and buf_queries.shape[0] > 0:
                                if self.use_bbox:
                                    rel_dist_buf = self.iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(buf_bboxes))
                                    rel_dist_buf = rel_dist_buf.unsqueeze(-1)
                                else:
                                    det_pos = det_bboxes[:, :3]
                                    buf_pos = buf_bboxes[:, :3]
                                    rel_dist_buf = (det_pos[:, None] - buf_pos[None]) ** 2
                                    rel_dist_buf = rel_dist_buf.sum(-1, keepdim=True).sqrt()
                                geometry_embedding_buf = self.rel_dist_embed(rel_dist_buf)
                                appear_embedding_buf = obj_embedding[:, None] * buf_queries[None]
                                fused_embedding_buf = appear_embedding_buf + geometry_embedding_buf
                                det2buffer_mat = self.embed_trans(fused_embedding_buf).sum(-1)
                                det2buffer_mat = det2buffer_mat.softmax(1)
                                if self.use_relative_asso:
                                    det2buffer_heatmap = self.embed_trans2(fused_embedding_buf).sum(-1).sigmoid()
                                    det2buffer_mat = det2buffer_mat * det2buffer_heatmap

                    else:
                        assert len(valid_det) > 0
                        abs_boxes = x['bboxes'][batch_idx][valid_det]
                        abs_boxes[:, :3] += sp_xyz[batch_idx][valid_det][:, :3] # 这里是将sp_xyz的偏移量加到abs_boxes上
                        track_instances.bboxes[batch_idx][:valid_det.shape[0], :] = abs_boxes
                        track_instances.valid_track[batch_idx][:valid_det.shape[0]] = True
                        track_instances.long_track[batch_idx][:valid_det.shape[0]] = False
                        track_instances.active[batch_idx][:valid_det.shape[0]] = True
                        track_instances.disappear_time[batch_idx][:valid_det.shape[0]] = 0
                        track_instances.obj_idxes[batch_idx][:valid_det.shape[0]] = torch.arange(valid_det.shape[0], device=valid_det.device)
                        track_instances.track_age[batch_idx][:valid_det.shape[0]] = 0
                        track_instances.queries[batch_idx][:valid_det.shape[0], :] = track_embedding_for_update
                        track_instances.obj_labels[batch_idx][:valid_det.shape[0]] = 0
                        # Store a scalar det score per track.
                        # CA (num_classes==1): prob[:, :-1] is (N,1) -> flatten to (N,)
                        # CS (num_classes>1): use max non-bg prob as score.
                        _prob = F.softmax(x['cls_preds'][batch_idx][valid_det], dim=-1)
                        if _prob.dim() == 2 and _prob.shape[1] > 2:
                            _score = _prob[:, :-1].max(dim=1).values
                        else:
                            _score = _prob[:, :-1].flatten(0, 1)
                        track_instances.scores[batch_idx][:valid_det.shape[0]] = _score
                        track_instances.global_track_id[batch_idx][:valid_det.shape[0]] = torch.arange(valid_det.shape[0], device=valid_det.device)
                        track_instances.category[batch_idx][:valid_det.shape[0]] = det_category
                        self.current_max_track_id = valid_det.shape[0]
                        det2track_mat = None
                        # Scheme A: compute det2buffer_mat when there are buffered tracks
                        det2buffer_mat = None
                        if 'online_merger' in locals() and getattr(online_merger, 'use_buffer', False):
                            buf_ids, buf_queries, buf_bboxes, buf_cats = online_merger.get_buffer_snapshots(device=det_embedding.device)
                            if buf_queries is not None and buf_queries.shape[0] > 0:
                                if self.use_bbox:
                                    rel_dist_buf = self.iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(buf_bboxes))
                                    rel_dist_buf = rel_dist_buf.unsqueeze(-1)
                                else:
                                    det_pos = det_bboxes[:, :3]
                                    buf_pos = buf_bboxes[:, :3]
                                    rel_dist_buf = (det_pos[:, None] - buf_pos[None]) ** 2
                                    rel_dist_buf = rel_dist_buf.sum(-1, keepdim=True).sqrt()
                                geometry_embedding_buf = self.rel_dist_embed(rel_dist_buf)
                                appear_embedding_buf = obj_embedding[:, None] * buf_queries[None]
                                fused_embedding_buf = appear_embedding_buf + geometry_embedding_buf
                                det2buffer_mat = self.embed_trans(fused_embedding_buf).sum(-1)
                                det2buffer_mat = det2buffer_mat.softmax(1)
                                if self.use_relative_asso:
                                    det2buffer_heatmap = self.embed_trans2(fused_embedding_buf).sum(-1).sigmoid()
                                    det2buffer_mat = det2buffer_mat * det2buffer_heatmap
                        det2buffer_mat = None

                else:
                    raise NotImplementedError(f"Unknown mot_type: {self.mot_type}")

            ## Query projector, semantic and geometric information
            if hasattr(self, 'merge_head'): # True
                query_feats = self.merge_head(x['queries'][0])
                query_feats_list.append([query_feats[mapping[0]], query_feats[mapping[1]]]) # 取出对应的query
                sem_preds = x['cls_preds'][0]
                sem_preds_list.append([sem_preds[mapping[0]], sem_preds[mapping[1]]]) # 前景背景
                sp_xyz_list.append([sp_xyz[0][mapping[0]], sp_xyz[0][mapping[1]]])
                if self.use_bbox:
                    bbox_preds = x['bboxes'][0] # [N, 6]
                    bboxes_list.append([bbox_preds[mapping[0]], bbox_preds[mapping[1]]])
            ## Online merging
            if self.test_cfg.merge_type == 'learnable_online': # True

                if frame_i == 0:
                    diag_cfg = {}
                    if online_monitor_enable and isinstance(online_monitor_cfg, dict):
                        diag_cfg = online_monitor_cfg.get("bbox_center_diag", {}) or {}
                    if self.use_mot and self.mot_type == 'dq_track':
                        online_merger = DQ_Track_OnlineMerge(
                            self.test_cfg.inscat_topk_insts,
                            self.use_bbox,
                            self.asso_config.get('update_type', 'count'),
                            self.asso_config.get('use_buffer', False),
                            diag_cfg=diag_cfg)
                    else:
                        online_merger = OnlineMerge(
                            self.test_cfg.inscat_topk_insts,
                            self.use_bbox,
                            diag_cfg=diag_cfg)
                online_merger_ref = online_merger
                if self.use_mot and self.mot_type == 'dq_track':
                    mv_mask, mv_labels, mv_scores, self.current_max_track_id= online_merger.merge( # , mv_queries
                        results[-1].pop('pts_instance_mask')[0],
                        results[-1].pop('instance_labels')[0],
                        results[-1].pop('instance_scores')[0],
                        results[-1].pop('instance_queries')[0],
                        query_feats_list.pop(-1)[0],
                        sem_preds_list.pop(-1)[0],
                        sp_xyz_list.pop(-1)[0],
                        bboxes_list.pop(-1)[0] if self.use_bbox else None,
                        det2track_mat, det2buffer_mat, track_instances, track_embedding_for_update, self.current_max_track_id, det_category,
                        gt_inst=(self._monitor_frame_ctx.get("gt_inst", None) if isinstance(getattr(self, "_monitor_frame_ctx", None), dict) else None),
                        frame_idx=int(frame_i),
                        frame_points_xyz=(batch_inputs_dict['points'][0][frame_i, :, :3] if isinstance(batch_inputs_dict.get('points', None), (list, tuple)) else None))
                else:
                    mv_mask, mv_labels, mv_scores, mv_bboxes = online_merger.merge( # , mv_queries
                        results[-1].pop('pts_instance_mask')[0],
                        results[-1].pop('instance_labels')[0],
                        results[-1].pop('instance_scores')[0],
                        results[-1].pop('instance_queries')[0],
                        query_feats_list.pop(-1)[0],
                        sem_preds_list.pop(-1)[0],
                        sp_xyz_list.pop(-1)[0],
                        bboxes_list.pop(-1)[0] if self.use_bbox else None,)
                if self.use_mot and self.mot_type == 'dq_track':
                    pass
                    # track_instances = self.update_track_instances_predict(track_instances, mapping, indices, is_last)
                # Empty cache. Only offline merging requires the whole list.
                torch.cuda.empty_cache()
                if online_monitor_enable and online_monitor is not None:
                    fr = {"frame": int(frame_i)}
                    # Fragment merge stats (LMI/SCL) from merge_superpixels_extract_feat
                    if isinstance(getattr(self, "_last_sp_merge_stats", None), dict):
                        fr["sp_merge"] = self._last_sp_merge_stats
                    # Det filter stats from predict_by_feat_instance
                    if isinstance(getattr(self, "_last_det_filter_stats", None), dict):
                        fr["det_filter"] = self._last_det_filter_stats
                    # GDINO 2D-3D alignment diagnostics
                    if isinstance(getattr(self, "_last_gdino_diag_stats", None), dict):
                        fr["gdino"] = self._last_gdino_diag_stats
                    # GDINO DACA-2D diagnostics (full forward + query2d_pos)
                    if isinstance(getattr(self, "_last_gdino_daca2d_stats", None), dict):
                        fr["gdino_daca2d"] = self._last_gdino_daca2d_stats
                    # GDINO per-point 2D feature fusion diagnostics (image-level srcs -> point feats)
                    if isinstance(getattr(self, "_last_gdino_point_fusion_stats", None), dict):
                        fr["gdino_point_fusion"] = self._last_gdino_point_fusion_stats
                    # Decoder-side DACA-2D apply stats (gate sparsity; helps diagnose whether injection is near no-op).
                    try:
                        st = getattr(self.decoder, '_last_daca2d_apply_stats', None)
                        if isinstance(st, dict):
                            agg = st.get('agg', None) if isinstance(st.get('agg', None), dict) else None
                            if isinstance(agg, dict):
                                nq2d = float(agg.get('nq2d_mean', 0.0) or 0.0)
                                allowed_mean = float(agg.get('allowed_q2d_mean', 0.0) or 0.0)
                                fr['daca2d_apply'] = {
                                    'allowed_zero_rate': float(agg.get('allowed_q2d_zero_rate', 0.0) or 0.0),
                                    'allowed_q2d_mean': allowed_mean,
                                    'allowed_q2d_p50': float(agg.get('allowed_q2d_p50', 0.0) or 0.0),
                                    'allowed_q2d_p90': float(agg.get('allowed_q2d_p90', 0.0) or 0.0),
                                    'nq2d': nq2d,
                                    'nq3d': float(agg.get('nq3d_mean', 0.0) or 0.0),
                                    'density': float(allowed_mean / (nq2d + 1e-6)) if nq2d > 0 else 0.0,
                                    'delta_rel_mean': float(agg.get('delta_rel_mean', 0.0) or 0.0),
                                    'q2d_any_sp_rate': float(agg.get('q2d_any_sp_rate', 0.0) or 0.0),
                                }
                    except Exception:
                        pass
                    # Decoder-side track-window STM apply stats.
                    try:
                        trk_stm_mon = online_monitor_cfg.get("trk_stm", {}) if isinstance(online_monitor_cfg, dict) else {}
                        if bool(trk_stm_mon.get("enable", False)):
                            st = getattr(self.decoder, "_last_trk_stm_apply_stats", None)
                            if isinstance(st, dict):
                                fr["trk_stm_apply"] = st
                    except Exception:
                        pass
                    # GT-aligned embedding stability stats (oracle association by GT IoU).
                    if isinstance(gt_emb_diag_by_frame.get(int(frame_i), None), dict):
                        fr["gt_emb_diag"] = gt_emb_diag_by_frame[int(frame_i)]
                    # Online association stats from merger
                    if isinstance(getattr(online_merger, "last_stats", None), dict):
                        fr["assoc"] = online_merger.last_stats
                    online_monitor["frames"].append(fr)
        
        ## Offline merging
        if self.test_cfg.merge_type == 'learnable':
            mv_mask, mv_labels, mv_scores = ins_merge_mat(
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                [res['instance_queries'][0] for res in results],
                [res[0] for res in query_feats_list],
                [res[0] for res in sem_preds_list],
                [res[0] for res in sp_xyz_list],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_merge_mat(
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                [res['instance_queries'][1] for res in results],
                [res[1] for res in query_feats_list],
                [res[1] for res in sem_preds_list],
                [res[1] for res in sp_xyz_list],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'concat':
            mv_mask, mv_labels, mv_scores = ins_cat(
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_cat(
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'geometric':
            mv_mask, mv_labels, mv_scores = ins_merge(
                [points for points in batch_inputs_dict['points'][0]],
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                [res['instance_queries'][0] for res in results],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_merge(
                [points for points in batch_inputs_dict['points'][0]],
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                [res['instance_queries'][1] for res in results],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'learnable_online':
            pass
        else:
            raise NotImplementedError("Unknown merge_type.")

        ## Offline panoptic segmentation
        mv_sem = torch.cat([res['pts_semantic_mask'][0] for res in results])
        
        # if self.use_bbox and not self.use_mot:
        #     batch_data_samples[0].pred_bbox = mv_bboxes.cpu().numpy()
        
        # Not mapping to reconstructed point clouds, return directly for visualization
        if not self.map_to_rec_pcd: # False
            merged_result = PointData(
                pts_semantic_mask=[mv_sem.cpu().numpy()],
                pts_instance_mask=[mv_mask.cpu().numpy()],
                instance_labels=mv_labels.cpu().numpy(),
                instance_scores=mv_scores.cpu().numpy())
            batch_data_samples[0].pred_pts_seg = merged_result
            return batch_data_samples
        
        ## Mapping to reconstructed point clouds for evaluation
        mv_xyz = batch_inputs_dict['points'][0][:, :, :3].reshape(-1, 3)
        rec_xyz = torch.tensor(batch_data_samples[0].eval_ann_info['rec_xyz'])[:, :3]
        target_coord = rec_xyz.to(mv_xyz.device).contiguous().float() # [239388, 3]
        target_offset = torch.tensor(target_coord.shape[0]).to(mv_xyz.device).float()
        source_coord = mv_xyz.contiguous().float() # [680000, 3]
        source_offset = torch.tensor(source_coord.shape[0]).to(mv_xyz.device).float()
        indices, dis = pointops.knn_query(1, source_coord, source_offset, target_coord, target_offset)
        indices = indices.reshape(-1).long()

        merged_result = PointData(
            pts_semantic_mask=[mv_sem[indices].cpu().numpy()],
            pts_instance_mask=[mv_mask[:, indices].cpu().numpy()],
            instance_labels=mv_labels.cpu().numpy(),
            instance_scores=mv_scores.cpu().numpy())

        # Ensemble the predictions with mesh segments (eval_ann_info['segment_ids']) 
        if 'segment_ids' in batch_data_samples[0].eval_ann_info: # True
            merged_result = self.segment_smooth(merged_result, mv_xyz.device,
                batch_data_samples[0].eval_ann_info['segment_ids'])
        batch_data_samples[0].pred_pts_seg = merged_result
        if online_monitor_enable and online_monitor is not None:
            merged_result.online_monitor = online_monitor
        # Optional: export bbox/center diagnostics as a separate payload (not part of online_monitor JSON).
        try:
            if online_monitor_enable and isinstance(online_monitor_cfg, dict):
                diag_cfg = online_monitor_cfg.get("bbox_center_diag", {}) or {}
                if bool(diag_cfg.get("enable", False)) and isinstance(locals().get("online_merger_ref", None), DQ_Track_OnlineMerge):
                    merged_result.bbox_center_diag = online_merger_ref.export_bbox_center_diag()
        except Exception:
            pass
        # Clean online merger state after exporting optional diagnostics.
        try:
            if self.test_cfg.merge_type == 'learnable_online':
                om = locals().get("online_merger_ref", None)
                if om is not None and hasattr(om, "clean"):
                    om.clean()
        except Exception:
            pass

        return batch_data_samples
    def merge_superpixels(self, current_sp_pts_mask, current_pt_instance_mask, overlap_threshold=0.8):

        merge_masks = []

        for batch_idx in range(len(current_sp_pts_mask)):
            sp_pts_mask = current_sp_pts_mask[batch_idx]
            pts_instance_mask = current_pt_instance_mask[batch_idx]

            merged_mask = sp_pts_mask.clone()

            merged_groups = []

            for inst_id in pts_instance_mask.unique():
                if inst_id == -1: 
                    continue
                inst_mask = (pts_instance_mask == inst_id)
                if not inst_mask.any():
                    continue
                sp_ids_in_instance = sp_pts_mask[inst_mask].unique() #
                valid_sps = []
                for sp_id in sp_ids_in_instance:
                    sp_mask = (sp_pts_mask == sp_id) 
                
                    overlap_ratio = (sp_mask & inst_mask).sum().float() / sp_mask.sum().float()
                    if overlap_ratio > overlap_threshold:
                        valid_sps.append(sp_id.item())
                        
                if len(valid_sps) > 1:
                    merged_groups.append(valid_sps)
                    for sp_id in valid_sps:
                        merged_mask[sp_pts_mask == sp_id] = valid_sps[0] 
            all_original_ids = sp_pts_mask.unique().tolist()
            merged_ids = set()
            for group in merged_groups:
                merged_ids.update(group)
            unmerged_ids = [id for id in all_original_ids if id not in merged_ids]
            
            new_id = 0
            id_mapping = {}
            
            for group in merged_groups:
                for old_id in group:
                    id_mapping[old_id] = new_id
                new_id += 1
                
            for old_id in unmerged_ids:
                id_mapping[old_id] = new_id
                new_id += 1
            
            final_mask = merged_mask.clone()
            for old_id, new_id in id_mapping.items():
                final_mask[merged_mask == old_id] = new_id

            
            merge_masks.append(final_mask)
            # merged_sp_masks.append(merged_sp_mask)
        
        return merge_masks #, merged_sp_masks


    def merge_superpixels_predict(self, batch_inputs_dict, batch_data_samples, frame_i, current_sp_pts_mask, current_pt_instance_mask, overlap_threshold=0.99):
        merge_masks = []
        merged_sp_masks = []
        merged_groups_list = []
        class_names = [
            'cabinet', 'bed', 'chair', 'sofa', 'table',
            'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
            'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
            'bathtub', 'otherfurniture'
        ]
        id2class = dict(enumerate(class_names))
        for batch_idx in range(len(current_sp_pts_mask)):
            labels_3d = batch_data_samples[batch_idx].gt_instances_3d.labels_3d[frame_i]
            sp_pts_mask = current_sp_pts_mask[batch_idx]
            pts_instance_mask = current_pt_instance_mask[batch_idx]
            #sp_masks = current_sp_masks[batch_idx]  # [gt_num+201, num_queries]
            
            merged_mask = sp_pts_mask.clone()
            
            merged_groups = [] 
            
            for inst_id in pts_instance_mask.unique():
                
                if inst_id == -1: # -1号是背景
                    continue
                category = labels_3d[inst_id]
                assert category != -1, f"Invalid category for instance {inst_id}."
                # if category == 6 or category == 15: # 6: table, 7: chair
                #     continue
                inst_mask = (pts_instance_mask == inst_id)
                if not inst_mask.any():
                    continue
                sp_ids_in_instance = sp_pts_mask[inst_mask].unique() 
                valid_sps = []
                for sp_id in sp_ids_in_instance:
                    # if sp_id == -1: 
                    #     continue
                    sp_mask = (sp_pts_mask == sp_id) 
                
                    # if (sp_mask & inst_mask).sum() == sp_mask.sum():
                    #     valid_sps.append(sp_id.item())
                    
                    overlap_ratio = (sp_mask & inst_mask).sum().float() / sp_mask.sum().float()
                    if overlap_ratio > overlap_threshold:
                        valid_sps.append(sp_id.item())
                        
                if len(valid_sps) > 1:
                    merged_groups.append(valid_sps)
                    # print(id2class[category.item()],': valid_sps:', valid_sps)
                    for sp_id in valid_sps:
                        merged_mask[sp_pts_mask == sp_id] = valid_sps[0]

            all_original_ids = sp_pts_mask.unique().tolist()
            merged_ids = set()
            for group in merged_groups:
                merged_ids.update(group)
            unmerged_ids = [id for id in all_original_ids if id not in merged_ids]

            new_id = 0
            id_mapping = {}
            
            for group in merged_groups:
                for old_id in group:
                    id_mapping[old_id] = new_id
                new_id += 1
                
            for old_id in unmerged_ids:
                id_mapping[old_id] = new_id
                new_id += 1
            
            final_mask = merged_mask.clone()
            for old_id, new_id in id_mapping.items():
                final_mask[merged_mask == old_id] = new_id
            if len(merged_groups) > 0:
                merged_sp_masks.append(build_pairwise_mask(merged_groups, compact=False, max_value=sp_pts_mask.max()))
                merge_masks.append(final_mask)
                merged_groups_list.append(merged_groups)
            else:
                merged_sp_masks.append(([], []))
                merge_masks.append(final_mask)
                merged_groups_list.append([])
        if isinstance(self, ScanNet200MixFormer3D_FF_Online):
            with torch.no_grad():
                img_features = []
                for img_paths in batch_inputs_dict['img_paths']:
                    img_features.append(self.img_backbone(img_paths[frame_i])[0])
            img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]
            for img_meta in img_metas:
                img_meta['depth2img'] = img_meta['depth2img'][frame_i]
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict: # False
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates # [20000, 3]

        coordinates, features = ME.utils.batch_sparse_collate( # [20000, 4] [20000, 3]
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features) 

        # forward of backbone and neck 
        if isinstance(self, ScanNet200MixFormer3D_FF_Online):
            x = self.backbone(field.sparse(),
                            partial(self._f, img_features=img_features, img_metas=img_metas, img_shape=img_metas[0]['img_shape']),
                            memory=self.memory if hasattr(self,'memory') else None)
        else:
            x = self.backbone(field.sparse(), memory=self.memory if hasattr(self,'memory') else None) # [13141, 96]
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field) # [20000, 96]
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)] # [20000, 99] 
        x = x.features # [20000, 96]

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        if self.use_temporal_loss and self.inst_dict is not None:
            best_obj_ids_list = []
        for batch_idx, (data_sample, tmp_xyz) in enumerate(zip(batch_data_samples, all_xyz)):
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i].clone() # [20000] 
            sp_pts_mask = sp_pts_mask.to(dtype=torch.long)
            _, sp_pts_mask = torch.unique(sp_pts_mask, sorted=True, return_inverse=True)
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points)) # [20000] 
            n_super_points.append(int(sp_pts_mask.max().item()) + 1 if sp_pts_mask.numel() else 0)
        sp_idx = torch.cat(sp_pts_masks) # [20000]
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz, with_xyz=True) # [N_segment, 96], [20000, 1]
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        # super_points = ([bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples], all_xyz_w) # ([20000], [20000, 1])
        # x_final = self.decoder(features, point_features, features, super_points)
        x_detach = [features[i] for i in range(len(features))]
        pred_bboxes = []
        pred_cls_list = []
        new_queries = []
        queries = self.decoder._get_queries(x_detach, len(current_sp_pts_mask)) 
        for i in range(len(queries)):
            norm_query = self.decoder.out_norm(queries[i])
            reg_final = self.decoder.out_reg(norm_query) # [N_segments, 256] -> [N_segments, 6]
            reg_cls = self.decoder.out_cls(norm_query) # [N_segments, 256] -> [N_segments, 1]
            reg_cls = reg_cls.softmax(1)
            
            reg_distance = torch.exp(reg_final[:, 3:6])
            pred_bbox = torch.cat([reg_final[:, :3], reg_distance], dim=1)

            pred_cls_list.append(reg_cls)
            pred_bboxes.append(pred_bbox)
            new_queries.append(norm_query)
        x_detach = [new_queries[i] for i in range(len(new_queries))]
        valid_sps_list = []
        for batch_idx in range(len(pred_bboxes)):
            # invalid_index = pred_cls_list[batch_idx][:, 0] < 0.5
            labels = pred_cls_list[batch_idx].argmax(dim=1) 
            bg = pred_cls_list[batch_idx].shape[1] - 1
            labels_mask = ((labels[:, None] == labels)  & (labels[:, None] != bg) & (labels != bg)[:, None]  ) # 
            det_bboxes = pred_bboxes[batch_idx].clone() # [N_det, 6]
            det_bboxes[:, :3] += sp_xyz_list[batch_idx][:, :3]
            pos_embedding = self.merge_box_trans(det_bboxes)
            obj_embedding1, obj_embedding2 = self.merge_query_inter(x_detach[batch_idx], x_detach[batch_idx], pos_embedding)
            merge_rel_dist = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes))
            merge_rel_dist = merge_rel_dist.unsqueeze(-1) # [N_det, N_det, 1]
            merge_geometry_embedding = self.merge_dist_embed(merge_rel_dist) # [N_det, N_det, 1] -> [N_det, N_det, 256]
            merge_appear_embedding = obj_embedding1[:,None] * obj_embedding2[None] # [N_det, N_det, 256]
            # V1
            # merge_fused_embedding = merge_appear_embedding + merge_geometry_embedding
            # V2
            # fused = torch.cat([merge_appear_embedding, merge_geometry_embedding], dim=-1)
            # merge_fused_embedding = self.fuse_linear(fused)  # nn.Linear(2*D, D)
            # V3
            merge_fused_embedding = self.merge_fusion(merge_appear_embedding, merge_geometry_embedding)
            merge_det_mat = self.merge_embed_trans(merge_fused_embedding).sum(-1)
            m = merge_det_mat.sigmoid() # [N_det, N_det]

            # m = merge_det_mat.softmax(dim=1)    
            m_sym = (m + m.t()) / 2             
            if len(merged_sp_masks[batch_idx][0]) > 0:
                merge_det_heatmap = merged_sp_masks[batch_idx][0].to(det_bboxes.device).float()
            else:
                merge_det_heatmap = torch.zeros_like(m_sym).to(det_bboxes.device).float()
            # cluster = cluster_with_threshold(m_sym, 0.7)
            # cluster = cluster_with_threshold(self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes), mode='giou'), 0.7)
            # print('===================================================================')
            tmp_heatmap = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes))
            error = torch.abs(tmp_heatmap - merge_det_heatmap).sum() / len(det_bboxes) 
            tmp_heatmap_mask = (tmp_heatmap > 0.5).float()
            
            error_mask = torch.abs(tmp_heatmap_mask - merge_det_heatmap).sum() / len(det_bboxes) 
            # print(f"Error: {error.item()}, Error mask{error_mask.item()}, Num Det Bboxes: {len(det_bboxes)}")
            best_threshold, best_error_mask = find_optimal_threshold(tmp_heatmap, merge_det_heatmap, det_bboxes)
            # print(f"Best threshold: {best_threshold}, Best error mask: {best_error_mask.item()}")
            if len(det_bboxes) in self.acc_dict:
                self.acc_dict[len(det_bboxes)].append(best_threshold)
            else:
                self.acc_dict[len(det_bboxes)] = [best_threshold]
            
            # error2 = torch.abs(m_sym - merge_det_heatmap).sum() / len(det_bboxes)
            # tmp_heatmap_mask2 = (m_sym > 0.5).float()
            # error2_mask = torch.abs(tmp_heatmap_mask2 - merge_det_heatmap).sum() / len(det_bboxes)
            # print(f"Error2: {error2.item()}, Error mask{error2_mask.item()}, Num Det Bboxes: {len(det_bboxes)}")
            # best_threshold2, best_error_mask2 = find_optimal_threshold(m_sym, merge_det_heatmap, det_bboxes)
            # print(f"Best threshold2: {best_threshold2}, Best error mask2: {best_error_mask2.item()}")

            # iou_mask = tmp_heatmap < 0.2
            # m_sym[iou_mask] = 0
            # error3 = torch.abs(m_sym - merge_det_heatmap).sum() / len(det_bboxes)
            # tmp_heatmap_mask3 = (m_sym > 0.5).float()
            # error3_mask = torch.abs(tmp_heatmap_mask3 - merge_det_heatmap).sum() / len(det_bboxes)
            # print(f"Error3: {error3.item()}, Error mask{error3_mask.item()}, Num Det Bboxes: {len(det_bboxes)}")
            # best_threshold3, best_error_mask3 = find_optimal_threshold(m_sym, merge_det_heatmap, det_bboxes)
            # print(f"Best threshold3: {best_threshold3}, Best error mask3: {best_error_mask3.item()}")
            # cluster = cluster_with_threshold(tmp_heatmap, best_threshold)
            # print('===================================================================')
            # if len(det_bboxes) > 100:
            #     cluster = cluster_with_threshold(tmp_heatmap, 0.65)
            # elif len(det_bboxes) > 50:
            #     cluster = cluster_with_threshold(tmp_heatmap, 0.65)
            # else:
            #     cluster = cluster_with_threshold(tmp_heatmap, 0.63)
            # cluster = cluster_with_threshold(tmp_heatmap, 0.63)
            iou_map = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes), mode='iou')
            # iou_map[invalid_index] = 0
            # iou_map[:, invalid_index] = 0
            # cluster_ori = cluster_with_threshold(m_sym, 0.95)
            # m_sym = torch.where(merge_rel_dist[:, :, 0] > 0.2, m_sym, 0)
            cluster_gt = cluster_complete_link(merge_det_heatmap, 0.96)
            # print(evaluate_clustering_pairwise(cluster_gt, cluster_ori))
            # cluster = cluster_complete_link(m_sym*labels_mask.float(), 0.9)
            # print(evaluate_clustering_pairwise(cluster_gt, cluster))

            cluster_iou = cluster_complete_link(iou_map*labels_mask.float(), 0.3)
            # print(evaluate_clustering_pairwise(cluster_gt, cluster_iou))
            # cluster_iou = cluster_complete_link(iou_map*labels_mask.float(), 0.4)
            # print(evaluate_clustering_pairwise(cluster_gt, cluster_iou))
            # cluster_iou = cluster_complete_link(iou_map*labels_mask.float(), 0.5)
            # print(evaluate_clustering_pairwise(cluster_gt, cluster_iou))
            # thresh_dict = {0: 0.4, 1: 0.5, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.4, 6: 0.5, 7: 0.4, 8: 0.5, 9: 0.5, 
            #                10: 0.3, 11: 0.5, 12: 0.5, 13: 0.5, 14: 0.3, 15: 0.4, 16: 0.5, 17: 0.3, 18: 0.5,}

            # clusters_per_cls = cluster_with_per_class_threshold(
            #     iou_map, labels_mask, labels,
            #     thresh_per_class=thresh_dict,
            #     base_thresh=1.0
            # )
            # print(evaluate_clustering_pairwise(cluster_gt, clusters_per_cls))
            cluster_both = cluster_complete_link(iou_map*m_sym*labels_mask.float(), 0.5)
            # print(evaluate_clustering_pairwise(cluster_gt, cluster_both))
            # print('==============================')
            
            valid_sps_list.append(cluster_both)
            # valid_sps_list.append(cluster_iou)
            # valid_sps_list.append(clusters_per_cls)
            # valid_sps_list.append(clusters_per_cls)
            

        new_merge_masks = []
        new_merged_sp_masks = []
        
        # 对每个batch进行处理
        for batch_idx in range(len(current_sp_pts_mask)):
            sp_pts_mask = current_sp_pts_mask[batch_idx]
            merged_mask = sp_pts_mask.clone()
            
            merged_groups = []
            for valid_sps in valid_sps_list[batch_idx]:
           
                if len(valid_sps) > 1:
                    merged_groups.append(valid_sps)
                    for sp_id in valid_sps:
                        merged_mask[sp_pts_mask == sp_id] = valid_sps[0] 

            all_original_ids = sp_pts_mask.unique().tolist()
            merged_ids = set()
            for group in merged_groups:
                merged_ids.update(group)
            unmerged_ids = [id for id in all_original_ids if id not in merged_ids]
            
            new_id = 0
            id_mapping = {}

            for group in merged_groups:
                for old_id in group:
                    id_mapping[old_id] = new_id
                new_id += 1
                
            for old_id in unmerged_ids:
                id_mapping[old_id] = new_id
                new_id += 1

            final_mask = merged_mask.clone()
            for old_id, new_id in id_mapping.items():
                final_mask[merged_mask == old_id] = new_id

            new_merged_sp_masks.append(merged_groups)
            new_merge_masks.append(final_mask)
        
        return new_merge_masks

    def merge_superpixels_train(self, batch_inputs_dict, batch_data_samples, frame_i, current_sp_pts_mask, current_pt_instance_mask, overlap_threshold=0.7):
        merge_masks = []
        merged_sp_masks = []
        
        for batch_idx in range(len(current_sp_pts_mask)):
            sp_pts_mask = current_sp_pts_mask[batch_idx]
            pts_instance_mask = current_pt_instance_mask[batch_idx]
            merged_mask = sp_pts_mask.clone()
            merged_groups = []
            for inst_id in pts_instance_mask.unique():
                if inst_id == -1:
                    continue
                inst_mask = (pts_instance_mask == inst_id)
                if not inst_mask.any():
                    continue
                sp_ids_in_instance = sp_pts_mask[inst_mask].unique() 
                valid_sps = []
                for sp_id in sp_ids_in_instance:

                    sp_mask = (sp_pts_mask == sp_id)

                    overlap_ratio = (sp_mask & inst_mask).sum().float() / sp_mask.sum().float()
                    if overlap_ratio > overlap_threshold:
                        valid_sps.append(sp_id.item())
                        
                if len(valid_sps) >= 1:
                    merged_groups.append(valid_sps)
                    for sp_id in valid_sps:
                        merged_mask[sp_pts_mask == sp_id] = valid_sps[0] 

            all_original_ids = sp_pts_mask.unique().tolist()
            merged_ids = set()
            for group in merged_groups:
                merged_ids.update(group)
            unmerged_ids = [id for id in all_original_ids if id not in merged_ids]

            new_id = 0
            id_mapping = {}
            
            for group in merged_groups:
                for old_id in group:
                    id_mapping[old_id] = new_id
                new_id += 1

            for old_id in unmerged_ids:
                id_mapping[old_id] = new_id
                new_id += 1

            final_mask = merged_mask.clone()
            for old_id, new_id in id_mapping.items():
                final_mask[merged_mask == old_id] = new_id

            merged_sp_masks.append(build_pairwise_mask(merged_groups, compact=True))
            merge_masks.append(final_mask)
        if isinstance(self, ScanNet200MixFormer3D_FF_Online):
            with frozen_inference(self.img_backbone):
                img_features = []
                for img_paths in batch_inputs_dict['img_paths']:
                    img_features.append(self.img_backbone(img_paths[frame_i])[0])
            img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]
            for img_meta in img_metas:
                img_meta['depth2img'] = img_meta['depth2img'][frame_i]
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict: # False
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates # [20000, 3]

        coordinates, features = ME.utils.batch_sparse_collate( # [20000, 4] [20000, 3]
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # Optional: GDINO sparse-FPN features for backbone fusion (keep consistent with extract_feat).
        # NOTE: this function is executed during training (merge loss); if `use_dino=True` and
        # `dino_strict=True`, we must guarantee that `dino_feats` is present, otherwise the
        # backbone will error at s8 fusion. Therefore we follow the same logic as `extract_feat`
        # and fail fast with a clear message when point-fusion is enabled but unavailable.
        use_gdino_fusion = bool(self.gdino_point_fusion_cfg.get("enable", False)) if isinstance(getattr(self, "gdino_point_fusion_cfg", None), dict) else False
        fuse_mode = str(self.gdino_point_fusion_cfg.get("fuse_mode", "concat")).lower() if isinstance(getattr(self, "gdino_point_fusion_cfg", None), dict) else "concat"
        pf_mode = str(self.gdino_point_fusion_cfg.get("mode", "early")).lower() if isinstance(getattr(self, "gdino_point_fusion_cfg", None), dict) else "early"
        if fuse_mode in ("fpn", "sparse_fpn"):
            pf_mode = "fpn"

        gdino_sparse_fpn = None
        if use_gdino_fusion and pf_mode == "fpn":
            gdino_point_feats = None
            gdino_point_stats = None
            try:
                gdino_point_feats, gdino_point_stats = self._run_gdino_point_fusion_for_frame(
                    batch_inputs_dict, batch_data_samples, int(frame_i)
                )
            except Exception as e:
                gdino_point_feats, gdino_point_stats = None, {"frame": int(frame_i), "error": repr(e)}
                if bool(self.gdino_point_fusion_cfg.get("log_fail", False)):
                    print(f"[GDINO][point_fusion][error] frame={int(frame_i)} err={repr(e)}")
            try:
                self._last_gdino_point_fusion_stats = gdino_point_stats
            except Exception:
                pass

            ok = isinstance(gdino_point_feats, (list, tuple)) and len(gdino_point_feats) == len(batch_inputs_dict.get("points", []))
            if not ok:
                raise RuntimeError(
                    "[GDINO][point_fusion] enabled but no per-point features were produced. "
                    f"stats={gdino_point_stats}"
                )
            if any(f is None or (not torch.is_tensor(f)) for f in gdino_point_feats):
                raise RuntimeError(
                    "[GDINO][point_fusion] enabled but got invalid per-point features. "
                    f"stats={gdino_point_stats}"
                )

            coords_list = []
            feats_list = []
            per_sample_lens = []
            for b in range(len(batch_inputs_dict.get("points", []))):
                if "elastic_coords" in batch_inputs_dict:
                    coord_src = batch_inputs_dict["elastic_coords"][b][frame_i] * self.voxel_size
                else:
                    coord_src = batch_inputs_dict["points"][b][frame_i, :, :3]
                coords = torch.floor(coord_src / self.voxel_size).to(dtype=torch.int32)
                feats_b = gdino_point_feats[b]
                if torch.is_tensor(feats_b) and feats_b.dim() > 2:
                    feats_b = feats_b.reshape(-1, feats_b.shape[-1])
                n_coords = int(coords.shape[0])
                n_feats = int(feats_b.shape[0]) if torch.is_tensor(feats_b) else -1
                per_sample_lens.append((b, n_coords, n_feats, tuple(feats_b.shape) if torch.is_tensor(feats_b) else None))
                if torch.is_tensor(feats_b) and n_feats != n_coords:
                    raise RuntimeError(
                        "[GDINO][point_fusion] merge_superpixels_train coords/feats length mismatch. "
                        f"frame={int(frame_i)} sample={b} n_coords={n_coords} n_feats={n_feats} feats_shape={tuple(feats_b.shape)} "
                        f"per_sample={per_sample_lens} stats={gdino_point_stats}"
                    )
                batch_col = torch.full((coords.shape[0], 1), b, dtype=torch.int32, device=coords.device)
                coords_batched = torch.cat([batch_col, coords], dim=1)
                coords_list.append(coords_batched)
                feats_list.append(feats_b.to(device=coords.device))
            coords_batch = torch.cat(coords_list, dim=0) if coords_list else None
            feats_batch = torch.cat(feats_list, dim=0) if feats_list else None
            if coords_batch is not None and feats_batch is not None:
                if int(coords_batch.shape[0]) != int(feats_batch.shape[0]):
                    raise RuntimeError(
                        "[GDINO][point_fusion] merge_superpixels_train batch coords/feats mismatch. "
                        f"frame={int(frame_i)} coords_batch={tuple(coords_batch.shape)} feats_batch={tuple(feats_batch.shape)} "
                        f"per_sample={per_sample_lens} stats={gdino_point_stats}"
                    )
                gdino_sparse_fpn = build_sparse_fpn(coords_batch, feats_batch)

        # forward of backbone and neck 
        if isinstance(self, ScanNet200MixFormer3D_FF_Online):
            with frozen_inference(self.backbone), frozen_inference(self.memory):
                x = self.backbone(field.sparse(),
                                partial(self._f, img_features=img_features, img_metas=img_metas, img_shape=img_metas[0]['img_shape']),
                                memory=self.memory if hasattr(self,'memory') else None)
        else:
            with frozen_inference(self.backbone), frozen_inference(self.memory):
                if getattr(self.backbone, "use_dino", False) and gdino_sparse_fpn is not None:
                    x = self.backbone(field.sparse(), dino_feats=gdino_sparse_fpn, memory=self.memory if hasattr(self,'memory') else None) # [13141, 96]
                else:
                    x = self.backbone(field.sparse(), memory=self.memory if hasattr(self,'memory') else None) # [13141, 96]
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field) # [20000, 96]
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)]
        x = x.features # [20000, 96]

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        if self.use_temporal_loss and self.inst_dict is not None:
            best_obj_ids_list = []
        for batch_idx, (data_sample, tmp_xyz) in enumerate(zip(batch_data_samples, all_xyz)):
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i].clone()
            sp_pts_mask = sp_pts_mask.to(dtype=torch.long)
            _, sp_pts_mask = torch.unique(sp_pts_mask, sorted=True, return_inverse=True)
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points)) 
            n_super_points.append(int(sp_pts_mask.max().item()) + 1 if sp_pts_mask.numel() else 0)
        sp_idx = torch.cat(sp_pts_masks) # [20000]
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz, with_xyz=True) # [N_segment, 96], [20000, 1]
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])

        
        x_detach = [features[i].detach() for i in range(len(features))]
        pred_bboxes = []
        new_queries = []
        with frozen_inference(self.decoder):
            queries = self.decoder._get_queries(x_detach, len(current_sp_pts_mask)) 
            for i in range(len(queries)):
                norm_query = self.decoder.out_norm(queries[i])
                reg_final = self.decoder.out_reg(norm_query) # [N_segments, 256] -> [N_segments, 6]
                reg_distance = torch.exp(reg_final[:, 3:6])
                pred_bbox = torch.cat([reg_final[:, :3], reg_distance], dim=1)
                pred_bboxes.append(pred_bbox)
                new_queries.append(norm_query)
        x_detach = [new_queries[i].detach() for i in range(len(new_queries))]
        pred_bboxes_detach = [pred_bboxes[i].detach() for i in range(len(pred_bboxes))]
        sp_xyz_list_detach = [sp_xyz_list[i].detach() for i in range(len(sp_xyz_list))]
        # loss = torch.tensor(0.0).to(x_detach[0].device)
        loss = self.get_mask_heatmap_loss(x_detach, pred_bboxes_detach, sp_xyz_list_detach, merged_sp_masks)
                    
        return loss
    def get_mask_heatmap_loss(self, x_detach, pred_bboxes_detach, sp_xyz_list,  merged_sp_masks):
        loss = 0
        for batch_idx in range(len(pred_bboxes_detach)):

            valid_det_idx = merged_sp_masks[batch_idx][1]
            if len(valid_det_idx) == 0:
                continue
            det_bboxes = pred_bboxes_detach[batch_idx][valid_det_idx].clone() # [N_det, 6]
            det_bboxes[:, :3] += sp_xyz_list[batch_idx][valid_det_idx][:, :3]
            pos_embedding = self.merge_box_trans(det_bboxes)
            obj_embedding1, obj_embedding2 = self.merge_query_inter(x_detach[batch_idx][valid_det_idx], x_detach[batch_idx][valid_det_idx], pos_embedding)
            merge_rel_dist = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes))
            merge_rel_dist = merge_rel_dist.unsqueeze(-1) # [N_det, N_det, 1]
            merge_geometry_embedding = self.merge_dist_embed(merge_rel_dist) # [N_det, N_det, 1] -> [N_det, N_det, 256]
            merge_appear_embedding = obj_embedding1[:,None] * obj_embedding2[None] # [N_det, N_det, 256]
            # V1
            # merge_fused_embedding = merge_appear_embedding + merge_geometry_embedding
            # V2
            # fused = torch.cat([merge_appear_embedding, merge_geometry_embedding], dim=-1)
            # merge_fused_embedding = self.fuse_linear(fused)  # nn.Linear(2*D, D)
            # V3
            merge_fused_embedding = self.merge_fusion(merge_appear_embedding, merge_geometry_embedding)
            merge_det_mat = self.merge_embed_trans(merge_fused_embedding).sum(-1)
            merge_det_mat = merge_det_mat.sigmoid() # [N_det, N_det]
            merge_det_heatmap = merged_sp_masks[batch_idx][0].to(det_bboxes.device).float()
            gamma = 2.0  
            alpha = 0.25  
            # loss += F.binary_cross_entropy(merge_det_mat, merge_det_heatmap, reduction='mean')
            bce_loss = F.binary_cross_entropy(merge_det_mat, merge_det_heatmap, reduction='none')
            # pt = torch.exp(-bce_loss)  

            # focal_loss = alpha * (1 - pt) ** gamma * bce_loss

            # 最终损失为加权后的Focal Loss
            loss += bce_loss.sum() / max(merge_det_heatmap.eq(1).float().sum().item(), 1)
            # loss += sigmoid_focal_loss(
            #     pred=merge_det_mat.view(-1),
            #     tgt=merge_det_heatmap.view(-1),
            #     alpha=0.25,
            #     gamma=2.0,
            #     reduction='mean'
            # )
        return loss
    def segment_smooth(self, results, device, segment_ids):
        # Robust to empty instance predictions: torch_scatter.scatter_mean
        # will crash when src has a zero-size dimension.
        if segment_ids is None:
            return results

        segment_ids = np.asarray(segment_ids)
        if segment_ids.size == 0:
            return results

        sem_np = results.pts_semantic_mask[0]
        ins_np = results.pts_instance_mask[0]
        if sem_np is None or ins_np is None:
            return results

        sem_np = np.asarray(sem_np)
        ins_np = np.asarray(ins_np)

        num_points = int(sem_np.shape[0])
        if segment_ids.shape[0] != num_points:
            # Segment ids should align with reconstructed point count.
            return results
        if ins_np.ndim != 2 or ins_np.shape[1] != num_points:
            return results

        _, inverse = np.unique(segment_ids, return_inverse=True)

        def _run(run_device):
            seg_t = torch.as_tensor(inverse, device=run_device, dtype=torch.long)

            sem_t = torch.as_tensor(sem_np, device=run_device, dtype=torch.long)
            sem_seg = scatter_mean(F.one_hot(sem_t).float(), seg_t, dim=0)
            sem_out = sem_seg.argmax(dim=1)[seg_t]

            if ins_np.shape[0] == 0:
                results.pts_semantic_mask[0] = sem_out.cpu().numpy()
                return results

            ins_t = torch.as_tensor(ins_np, device=run_device).float()
            ins_seg = scatter_mean(ins_t, seg_t, dim=1)
            ins_out = (ins_seg > 0.5)[:, seg_t]

            results.pts_semantic_mask[0] = sem_out.cpu().numpy()
            results.pts_instance_mask[0] = ins_out.cpu().numpy()
            return results

        try:
            return _run(device)
        except RuntimeError as e:
            # Some MV scenes can be very large; if CUDA OOM happens here,
            # fall back to CPU to keep evaluation running.
            if getattr(device, 'type', None) == 'cuda' and 'out of memory' in str(e).lower():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return _run(torch.device('cpu'))
            raise
    
    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        inst_res = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.inst_score_thr) # dict [20000] 0.3
        sem_res = self.predict_by_feat_semantic(out, superpoints) # [20000]

        sem_map2 = self.predict_by_feat_semantic(
            out, superpoints, self.test_cfg.stuff_classes) # [20000]
        inst_res2 = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.pan_score_thr)

        pts_semantic_mask = [sem_res, sem_map2]
        pts_instance_mask = [inst_res[0].bool(), inst_res2[0].bool()]
        instance_labels = [inst_res[1], inst_res2[1]]
        instance_scores = [inst_res[2], inst_res2[2]]
        instance_queries = [inst_res[3], inst_res2[3]]
        mapping = [inst_res[4], inst_res2[4]]
      
        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=instance_labels,
                instance_scores=instance_scores,
                instance_queries=instance_queries)], mapping
    
    def predict_by_feat_instance(self, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        mapping = torch.arange(len(out['cls_preds'][0])).to(superpoints.device)
        cls_preds = out['cls_preds'][0] # [N_seg, 2]
        pred_masks = out['masks'][0] # [N_seg, 20000]
        queries = out['queries'][0] # [N_seg, 256]
        assert self.num_classes == 1 or self.num_classes == cls_preds.shape[1] - 1

        scores = F.softmax(cls_preds, dim=-1)[:, :-1] # [N_pred, 1] 
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        if self.num_classes == 1:
            scores = scores.sum(-1, keepdim=True)
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1) # N_pred
        topk_num = min(self.test_cfg.topk_insts, scores.shape[0] * scores.shape[1]) # 取最大前20
        scores, topk_idx = scores.flatten(0, 1).topk(topk_num, sorted=False) 
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        queries = queries[topk_idx] # [topk_num, 256]
        mapping = mapping[topk_idx] # [topk_num]

        if self.test_cfg.get('obj_normalization', None): 
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6) # [topk_num]
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel # 'linear'
            scores, labels, mask_pred_sigmoid, keep_inds = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel) 
            mapping = mapping[keep_inds]

        mask_pred_sigmoid = mask_pred_sigmoid[:, ...]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # Optional monitoring: record per-stage counts and GT-aware duplication stats.
        monitor_cfg = {}
        try:
            monitor_cfg = self.test_cfg.get('online_monitor', None) or {}
        except Exception:
            monitor_cfg = {}
        monitor_enable = bool(monitor_cfg.get('enable', False))
        det_stats = None
        if monitor_enable:
            det_stats = {
                "after_topk": int(topk_num),
                "after_nms": int(mask_pred.shape[0]),
                "sp_score_thr": float(self.test_cfg.get('sp_score_thr', 0.0)),
                "inst_score_thr": float(score_threshold),
                "npoint_thr": int(self.test_cfg.get('npoint_thr', 0)),
            }
            # GT-aware stats are computed sparsely to control runtime.
            ctx = getattr(self, "_monitor_frame_ctx", None)
            if isinstance(ctx, dict):
                gt_inst = ctx.get("gt_inst", None)
                stride = int(ctx.get("gt_frame_stride", 5))
                fr = int(ctx.get("frame", -1))
                do_gt = (gt_inst is not None) and (stride > 0) and (fr >= 0) and (fr % stride == 0)
                det_stats["gt_eval"] = bool(do_gt)
            else:
                det_stats["gt_eval"] = False

        # score_thr
        score_mask = scores > score_threshold # [n_preds] 
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]
        queries = queries[score_mask]
        mapping = mapping[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]
        queries = queries[npoint_mask]
        mapping = mapping[npoint_mask]

        if monitor_enable and det_stats is not None:
            det_stats["after_inst_thr"] = int(score_mask.sum().item())
            det_stats["after_npoint_thr"] = int(mask_pred.shape[0])
            # GT-aware duplication stats (hit0/hit_ge2) for pre-pool vs det_to_merge.
            ctx = getattr(self, "_monitor_frame_ctx", None)
            if isinstance(ctx, dict):
                gt_inst = ctx.get("gt_inst", None)
                stride = int(ctx.get("gt_frame_stride", 5))
                fr = int(ctx.get("frame", -1))
                do_gt = (gt_inst is not None) and (stride > 0) and (fr >= 0) and (fr % stride == 0)
                if do_gt:
                    try:
                        gt_vis_npoint = int(ctx.get("gt_vis_npoint", 100))
                        iou_thr = float(ctx.get("iou_thr", 0.5))
                        iou_lo = float(ctx.get("iou_lo_thr", 0.1))

                        # Reconstruct pre-pool masks after NMS+sp_thr but before inst/npoint filtering.
                        # Note: the tensors were already filtered by NMS; we stored them in `mask_pred_pre`.
                        # Here we recompute masks for pre-pool using the same intermediate variables.
                        # `mask_pred_pre` corresponds to `mask_pred` before score_mask was applied.
                        # We cannot reuse the overwritten `mask_pred`, so rebuild from cached `keep_inds` output.
                        # Simplest: treat `mask_pred_pre` as the binary masks right after sp_thr and before score_thr.
                        # It is still available as `mask_pred_sigmoid > sp_thr` with current keep_inds.
                        mask_pred_pre = (mask_pred_sigmoid > self.test_cfg.sp_score_thr)

                        def _build_gt_masks(gt_ids, vis_thr):
                            gt_ids = gt_ids.detach()
                            gt_ids = gt_ids.to(mask_pred_pre.device)
                            uniq = torch.unique(gt_ids)
                            uniq = uniq[uniq >= 0]
                            if uniq.numel() == 0:
                                return None
                            masks = []
                            for gid in uniq.tolist():
                                m = (gt_ids == int(gid))
                                if int(m.sum().item()) >= vis_thr:
                                    masks.append(m)
                            if not masks:
                                return None
                            return torch.stack(masks, dim=0)  # [G, P]

                        gt_masks = _build_gt_masks(gt_inst, gt_vis_npoint)
                        if gt_masks is None:
                            det_stats["gt_vis"] = 0
                        else:
                            det_stats["gt_vis"] = int(gt_masks.shape[0])

                            def _hit_stats(pred_masks, gt_masks, thr):
                                if pred_masks.numel() == 0 or gt_masks.numel() == 0:
                                    return {"hit0": 0.0, "hit_ge2": 0.0, "mean_mult": 0.0}
                                pm = pred_masks.float()
                                gm = gt_masks.float()
                                inter = pm @ gm.t()  # [P, G]
                                ps = pm.sum(1, keepdim=True)
                                gs = gm.sum(1, keepdim=True).t()
                                union = ps + gs - inter
                                iou = inter / (union + 1e-6)
                                hit = (iou >= thr).sum(0)  # [G]
                                hit0 = float((hit == 0).float().mean().item())
                                hit_ge2 = float((hit >= 2).float().mean().item())
                                pos = hit[hit >= 1].float()
                                mean_mult = float((pos.mean().item()) if pos.numel() else 0.0)
                                return {"hit0": hit0, "hit_ge2": hit_ge2, "mean_mult": mean_mult}

                            det_stats["pre_pool"] = {
                                "n_pred": int(mask_pred_pre.shape[0]),
                                "iou05": _hit_stats(mask_pred_pre, gt_masks, iou_thr),
                                "iou01": _hit_stats(mask_pred_pre, gt_masks, iou_lo),
                            }
                            det_stats["det_to_merge"] = {
                                "n_pred": int(mask_pred.shape[0]),
                                "iou05": _hit_stats(mask_pred, gt_masks, iou_thr),
                                "iou01": _hit_stats(mask_pred, gt_masks, iou_lo),
                            }
                    except Exception:
                        det_stats["gt_error"] = "gt_dup_stats_failed"
            try:
                self._last_det_filter_stats = det_stats
            except Exception:
                pass

        return mask_pred, labels, scores, queries, mapping
    
    def predict_by_feat_panoptic(self, sem_map, mask_pred, labels, scores):
        """Predict panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.test_cfg.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes, 
            mask_pred.shape[0] + n_stuff_classes, 
            device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask

        return sem_map, inst_map


@MODELS.register_module()
class ScanNet200MixFormer3D_FF_Online(ScanNet200MixFormer3D_Online):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 img_backbone=None,
                 backbone=None,
                 memory=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 merge_head=None,
                 merge_criterion=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,

                 use_query_memory=False,
                 use_self_attn=False,
                 use_noise=False,
                 noise_p=0.05,
                 noise_k=10,
                 use_temporal_loss=False,
                 use_decouple=False,
                 use_mot=False,
                 mot_type='motr',
                 train_asso_only=False,
                 matcher=None,
                 use_aug=False,
                 asso_loss_weight=0.5,
                 use_refine=False,
                 asso_config=None,
                 use_one2many=False,
                 criterion_one2many=None,
                 use_3d_refine=False,
                 reweight_dict=None,
                 use_relative_asso=False,
                 merge_sp_masks = False,
                 replace_bn_with_ln=False,
                 debug_mode=False
                 ):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.img_backbone = MODELS.build(img_backbone)
        self.backbone = MODELS.build(backbone)
        if memory is not None:
            self.memory = MODELS.build(memory)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.pool = MODELS.build(pool)
        self.decoder = MODELS.build(decoder)
        if merge_head is not None:
            self.merge_head = MODELS.build(merge_head)
        if merge_criterion is not None:
            self.merge_criterion = MODELS.build(merge_criterion)
        self.criterion = MODELS.build(criterion)
        self.decoder_online = decoder['temporal_attn']
        self.use_bbox = decoder['bbox_flag']
        self.sem_len = decoder['num_semantic_classes'] + 1 # 201
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.map_to_rec_pcd=True
        self.use_query_memory = use_query_memory
        if self.use_query_memory:
            self.muti_scale_query = MultiScaleQuery()
            self.query_memory = None
            self.pos_memory = None
            self.query_memory_relu = nn.ReLU()
            self.muti_scale_query.init_weights()
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.muti_scale_self_attn = MultiScaleQuery()
            self.self_attn_relu = nn.ReLU()
            self.muti_scale_self_attn.init_weights()
        self.use_noise = use_noise
        if self.use_noise:
            self.noise_p = noise_p
            self.noise_k = noise_k
        self.use_temporal_loss = use_temporal_loss
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_mask_memory = None
            self.before_sp_xyz = None
            self.before_query_ids = None
        self.use_decouple = use_decouple
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_query_boxes = None
        self.use_mot = use_mot
        if self.use_mot:
            self.mot_type = mot_type
            if mot_type == 'dq_track':
                self.use_relative_asso = use_relative_asso
                if self.use_relative_asso:
                    self.embed_trans2 = nn.Linear(256, 1)
                    self.heatmap_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
                self.use_refine = use_refine
                # self.asso_loss_weight = asso_loss_weight
                self.iou_calculator = AxisAlignedBboxOverlaps3D()
                self.matcher = TASK_UTILS.build(matcher)
                self.tracklet_trans = DQ_FFN(d_model=256, d_ffn=256, dropout=0)
                self.detector_trans = DQ_FFN(d_model=256, d_ffn=256, dropout=0)
                self.box_trans = nn.Sequential(
                    nn.Linear(6, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
                # query_trans = {'with_att': True, 'with_pos': True, 'min_channels': 256, 'drop_rate': 0.0}
                query_trans = asso_config['query_trans'] if asso_config is not None else {'with_att': True, 'with_pos': True, 'min_channels': 256, 'drop_rate': 0.0}
                self.query_inter = QueryInteractionX(in_channels=256, mid_channels=256, **query_trans)
                # self.update_type = asso_config['update_type'] if asso_config is not None else 'ema'
                self.asso_config = EasyDict(asso_config)
                
                self.rel_dist_embed = nn.Sequential(
                    nn.Linear(1, 256),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
                self.embed_trans = nn.Linear(256, 1)
                loss_asso = {'use_sigmoid': False, 'loss_weight': 1.0}
                from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
                self.loss_asso = CrossEntropyLoss(**loss_asso)
                self.ema_decay_rate = 0.5
                # self.train_asso_only = train_asso_only
            else:
                raise NotImplementedError(f"mot_type {mot_type} is not supported")

        self.use_one2many = use_one2many
        if self.use_one2many:
            self.one2many_loss_weight = 0.5
            self.criterion_one2many = MODELS.build(criterion_one2many[0])
        self.reweight_dict = reweight_dict
        self.merge_sp_masks = merge_sp_masks
        if self.merge_sp_masks:
            self.acc_dict = {}
            self.merge_box_trans = nn.Sequential(
                nn.Linear(6, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                DQ_FFN(d_model=256, d_ffn=256, dropout=0))
            self.merge_dist_embed = nn.Sequential(
                    nn.Linear(1, 256),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
            self.merge_embed_trans = nn.Linear(256, 1)
            self.merge_heatmap_loss = nn.BCEWithLogitsLoss(reduction='mean')
            self.merge_iou_calculator = AxisAlignedBboxOverlaps3D()
            self.fuse_linear = nn.Sequential(
                nn.Linear(512, 256, bias=False),
                nn.ReLU(inplace=True),
                nn.LayerNorm(256)
            )
            self.merge_fusion = MergeFusion(256)
            query_trans = {'with_att': True, 'with_pos': False, 'min_channels': 256, 'drop_rate': 0.0}
            self.merge_query_inter = QueryInteractionX(in_channels=256, mid_channels=256, **query_trans)
        self._prev_param_snapshot = None
        if replace_bn_with_ln:
            replace_bn(self)
        self.debug_mode = debug_mode
        self.init_weights()
        
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(960, 32, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True))
    
    def init_weights(self):
        if hasattr(self, 'memory'):
            self.memory.init_weights()
        if hasattr(self, 'img_backbone'):
            self.img_backbone.init_weights()
    def reset_query_memory(self):
        """Reset the detector.
        """
        if self.use_query_memory:
            self.query_memory = None
            self.pos_memory = None
    def extract_feat(self, batch_inputs_dict, batch_data_samples, frame_i):
        """Extract features from sparse tensor.
        """
        # extract image features
        with torch.no_grad():
            img_features = []
            for img_paths in batch_inputs_dict['img_paths']:
                img_features.append(self.img_backbone(img_paths[frame_i])[0])
        
        # TODO check
        img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]
        for img_meta in img_metas:
            img_meta['depth2img'] = img_meta['depth2img'][frame_i]
    
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            # pdb.set_trace()
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse(),
                          partial(self._f, img_features=img_features, img_metas=img_metas, img_shape=img_metas[0]['img_shape']),
                          memory=self.memory if hasattr(self,'memory') else None)
        if self.with_neck: # False
            x = self.neck(x)
        x = x.slice(field) # [45611, 96] -> [80000, 96]
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)]  # [B, N, 3+D]
        x = x.features # [80000, 96]

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i]
            sp_pts_mask = sp_pts_mask.to(dtype=torch.long)
            _, sp_pts_mask = torch.unique(sp_pts_mask, sorted=True, return_inverse=True)
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(int(sp_pts_mask.max().item()) + 1 if sp_pts_mask.numel() else 0)
        sp_idx = torch.cat(sp_pts_masks)
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz, with_xyz=True)

        # apply cls_layer
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        return features, point_features, all_xyz_w, sp_xyz_list # [N_segment, 96], [20000, 99], [20000, 1], [N_segment, 3]

    def _f(self, x, img_features, img_metas, img_shape):
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):
            coord_type = 'DEPTH'
            img_scale_factor = (
                point.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            #img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_flip = False
            img_crop_offset = (
                point.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_shape[-2:],
                img_shape=img_shape[-2:],
                aligned=True,
                padding_mode='zeros',
                align_corners=True))
 
        projected_features = torch.cat(projected_features, dim=0) # [N, 960]
        projected_features = ME.SparseTensor(
            projected_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        
        projected_features = self.conv(projected_features)
        return projected_features + x



class MultiScaleQuery(nn.Module):
    """Scale-adaptive Self Attention with variable-length tokens per batch."""
    def __init__(self, embed_dims=96, num_heads=8, dropout=0.1, init_cfg=None):

        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

        self.embed_dims = embed_dims
        self.num_heads = num_heads

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def forward(self, pos1_xyz, query_feat, key_feat, val_feat, pos2_xyz=None, dist_scale: float = 1.0):
        """
        pos1_xyz: list of [num_queries_i, 3]
        query_feat: list of [num_queries_i, embed_dims]
        key_feat: list of [num_keys_i, embed_dims]
        val_feat: list of [num_values_i, embed_dims]
        pos2_xyz: list of [num_keys_i, 3] or None
        """
        batch_size = len(query_feat)
        
        Q_max = max([q.shape[0] for q in query_feat])
        K_max = max([k.shape[0] for k in key_feat])
        V_max = max([v.shape[0] for v in val_feat])

        padded_query = torch.zeros(batch_size, Q_max, self.embed_dims, device=query_feat[0].device)
        padded_key = torch.zeros(batch_size, K_max, self.embed_dims, device=key_feat[0].device)
        padded_val = torch.zeros(batch_size, V_max, self.embed_dims, device=val_feat[0].device)
        
        query_mask = torch.zeros(batch_size, Q_max, dtype=torch.bool, device=query_feat[0].device)
        key_mask = torch.zeros(batch_size, K_max, dtype=torch.bool, device=key_feat[0].device)
        
        for i in range(batch_size):
            q_len = query_feat[i].shape[0]
            k_len = key_feat[i].shape[0]
            v_len = val_feat[i].shape[0]
            
            padded_query[i, :q_len, :] = query_feat[i]
            padded_key[i, :k_len, :] = key_feat[i]
            padded_val[i, :v_len, :] = val_feat[i]
            
            query_mask[i, :q_len] = 1
            key_mask[i, :k_len] = 1

        padded_pos1_xyz = torch.zeros(batch_size, Q_max, 3, device=pos1_xyz[0].device)
        pos1_mask_spatial = torch.zeros(batch_size, Q_max, dtype=torch.bool, device=pos1_xyz[0].device)
        for i in range(batch_size):
            q_len = pos1_xyz[i].shape[0]
            padded_pos1_xyz[i, :q_len, :] = pos1_xyz[i]
            pos1_mask_spatial[i, :q_len] = 1

        if pos2_xyz is not None:
            padded_pos2_xyz = torch.zeros(batch_size, K_max, 3, device=pos2_xyz[0].device)
            pos2_mask_spatial = torch.zeros(batch_size, K_max, dtype=torch.bool, device=pos2_xyz[0].device)
            for i in range(batch_size):
                k_len = pos2_xyz[i].shape[0]
                padded_pos2_xyz[i, :k_len, :] = pos2_xyz[i]
                pos2_mask_spatial[i, :k_len] = 1
        else:
            padded_pos2_xyz = None
            pos2_mask_spatial = None

        if pos2_xyz is None:
            dist = self.calc_bbox_dists(padded_pos1_xyz, pos1_mask_spatial)  # [B, Q_max, Q_max]
        else:
            dist = self.calc_bbox_dists2(padded_pos1_xyz, pos1_mask_spatial, 
                                        padded_pos2_xyz, pos2_mask_spatial)  # [B, Q_max, K_max]
        try:
            dist = dist * float(dist_scale)
        except Exception:
            pass
        
        if torch.isnan(dist).any():
            raise ValueError("NaN detected in dist")

        # 8. 生成 tau
        tau = self.gen_tau(padded_query)  # [B, Q_max, num_heads]
        tau = tau.clamp(max=10.0)  # Clamp to prevent extremely large values
        tau = tau.permute(0, 2, 1)  # [B, num_heads, Q_max]
        
        if torch.isnan(tau).any():
            raise ValueError("NaN detected in tau")

        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, num_heads, Q_max, K_max]

        if pos2_xyz is None:
            key_padding_mask = ~pos1_mask_spatial  # [B, Q_max], True 表示要屏蔽
        else:
            key_padding_mask = ~pos2_mask_spatial  # [B, K_max], True 表示要屏蔽

        attn_mask_combined = attn_mask.reshape(batch_size * self.num_heads, Q_max, -1)  # [B*num_heads, Q_max, K_max]

        assert attn_mask_combined.shape == (batch_size * self.num_heads, Q_max, attn_mask_combined.shape[-1]), "Attention mask shape mismatch"

        attn_output, attn_weights = self.attention(
            query=padded_query, 
            key=padded_key, 
            value=padded_val, 
            attn_mask=attn_mask_combined,
            key_padding_mask=key_padding_mask
        )
        
        if torch.isnan(attn_output).any():
            raise ValueError("NaN detected in attn_output")

        attn_output = attn_output * pos1_mask_spatial.unsqueeze(-1)  # [B, Q_max, D]

        queries = []
        for i in range(batch_size):
            queries.append(attn_output[i, :pos1_mask_spatial[i].sum(), :])
        
        return queries

    @torch.no_grad()
    def calc_bbox_dists(self, pos1_xyz, pos1_mask):
        pos1_xyz_exp1 = pos1_xyz.unsqueeze(2)  # [B, Q_max, 1, 3]
        pos1_xyz_exp2 = pos1_xyz.unsqueeze(1)  # [B, 1, Q_max, 3]

        dist = torch.norm(pos1_xyz_exp1 - pos1_xyz_exp2, dim=-1)  # [B, Q_max, Q_max]
        dist = -dist  # [B, Q_max, Q_max]

        return dist

    @torch.no_grad()
    def calc_bbox_dists2(self, pos1_xyz, pos1_mask, pos2_xyz, pos2_mask):
        pos1_xyz_exp = pos1_xyz.unsqueeze(2)  # [B, Q_max, 1, 3]
        pos2_xyz_exp = pos2_xyz.unsqueeze(1)  # [B, 1, K_max, 3]

        dist = torch.norm(pos1_xyz_exp - pos2_xyz_exp, dim=-1)  # [B, Q_max, K_max]

        dist = -dist  # [B, Q_max, K_max]

        return dist

def bbox_pred_to_bbox(bbox_pred):
    """Transform predicted bbox parameters to bbox.
    """
    if bbox_pred.shape[0] == 0:
        return bbox_pred
    bbox = bbox_pred

    return torch.stack(
        (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
            bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
            bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
        dim=-1)
