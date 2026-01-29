from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def _norm_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _get_bb_fields(bb_cfg: Optional[dict]) -> Tuple[str, str, str, str]:
    """Return (repo_dir, config_path, checkpoint, caption) for cache keying."""
    bb_cfg = bb_cfg if isinstance(bb_cfg, dict) else {}
    repo_dir = str(bb_cfg.get("repo_dir", ""))
    config_path = str(bb_cfg.get("config_path", ""))
    checkpoint = str(bb_cfg.get("checkpoint", ""))
    caption = str(bb_cfg.get("caption", ""))
    return repo_dir, config_path, checkpoint, caption


def gdino_cache_path(
    cache_dir: str,
    *,
    img_path: str,
    target_hw: Tuple[int, int],
    bb_cfg: Optional[dict],
    mode: str = "full",
) -> str:
    """Compute a deterministic cache file path.

    `mode` should be "full" (srcs + hs_last + pred_boxes + pred_scores) or
    "backbone" (srcs only). We include `mode` in the key to avoid accidental
    partial-data reads.
    """
    cache_dir = _norm_path(cache_dir)
    img_path_n = _norm_path(img_path)
    repo_dir, config_path, checkpoint, caption = _get_bb_fields(bb_cfg)
    # Visual features (srcs) do not depend on caption/text.
    # For `mode=backbone` we intentionally ignore `caption` so different prompt
    # configs can reuse the same cached `srcs`.
    if str(mode).lower() in ("backbone", "srcs"):
        caption = ""
    h, w = int(target_hw[0]), int(target_hw[1])

    key = "|".join(
        [
            "gdino_cache_v1",
            mode,
            img_path_n,
            f"{h}x{w}",
            repo_dir,
            config_path,
            checkpoint,
            caption,
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    # Two-level sharding keeps directories small.
    return os.path.join(cache_dir, mode, digest[:2], digest[2:4], f"{digest}.pt")


def _to_cpu_dtype(x: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"expected Tensor, got {type(x).__name__}")
    if x.device.type != "cpu":
        x = x.detach().to("cpu")
    else:
        x = x.detach()
    if x.dtype != dtype:
        x = x.to(dtype=dtype)
    return x.contiguous()


def save_gdino_cache(
    path: str,
    *,
    img_path: str,
    target_hw: Tuple[int, int],
    bb_cfg: Optional[dict],
    mode: str,
    out: Dict[str, Any],
    dtype: torch.dtype = torch.float16,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    repo_dir, config_path, checkpoint, caption = _get_bb_fields(bb_cfg)
    payload: Dict[str, Any] = {
        "version": 1,
        "mode": str(mode),
        "img_path": _norm_path(img_path),
        "target_hw": (int(target_hw[0]), int(target_hw[1])),
        "bb": {
            "repo_dir": repo_dir,
            "config_path": config_path,
            "checkpoint": checkpoint,
            "caption": caption,
        },
    }

    # srcs: list[(B,C,H,W)] or list[(C,H,W)]
    if "srcs" in out and isinstance(out["srcs"], list):
        srcs: List[torch.Tensor] = []
        for t in out["srcs"]:
            if not torch.is_tensor(t):
                continue
            if t.dim() == 4 and t.size(0) == 1:
                t = t.squeeze(0)
            srcs.append(_to_cpu_dtype(t, dtype=dtype))
        payload["srcs"] = srcs

    for k in ("hs_last", "pred_boxes", "pred_scores"):
        v = out.get(k, None)
        if torch.is_tensor(v):
            # full forward returns (B,...) for these fields; store without batch dim.
            if v.dim() >= 2 and v.size(0) == 1:
                v = v.squeeze(0)
            payload[k] = _to_cpu_dtype(v, dtype=dtype)

    torch.save(payload, path)


def _load_one(
    cache_dir: str,
    *,
    img_path: str,
    target_hw: Tuple[int, int],
    bb_cfg: Optional[dict],
    mode: str,
) -> Optional[Dict[str, Any]]:
    p = gdino_cache_path(
        cache_dir, img_path=img_path, target_hw=target_hw, bb_cfg=bb_cfg, mode=mode
    )
    if not os.path.exists(p):
        return None
    try:
        obj = torch.load(p, map_location="cpu")
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    # Basic sanity check.
    if int(obj.get("version", 0)) != 1:
        return None
    if str(obj.get("mode", "")) != str(mode):
        return None
    return obj


def load_gdino_cache_batched(
    cache_dir: str,
    *,
    img_paths: Sequence[str],
    target_hw: Tuple[int, int],
    bb_cfg: Optional[dict],
    mode: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[Dict[str, Any]]:
    """Load cached GDINO outputs and stack them into batch tensors.

    Returns None on any cache miss (all-or-nothing for determinism).
    """
    if not img_paths:
        return None
    items = []
    for p in img_paths:
        d = _load_one(cache_dir, img_path=p, target_hw=target_hw, bb_cfg=bb_cfg, mode=mode)
        if d is None:
            return None
        items.append(d)

    out: Dict[str, Any] = {}
    # srcs: list of levels, each stacked to (B,C,H,W)
    try:
        srcs0 = items[0].get("srcs", None)
        if isinstance(srcs0, list) and len(srcs0) > 0:
            nlv = len(srcs0)
            batched_srcs: List[torch.Tensor] = []
            for lv in range(nlv):
                lvl = []
                for it in items:
                    s = it.get("srcs", None)
                    if not isinstance(s, list) or lv >= len(s) or (not torch.is_tensor(s[lv])):
                        return None
                    t = s[lv]
                    if t.dim() == 3:
                        t = t.unsqueeze(0)
                    lvl.append(t)
                feat = torch.cat(lvl, dim=0).to(device=device)
                if dtype is not None and feat.dtype != dtype:
                    feat = feat.to(dtype=dtype)
                batched_srcs.append(feat)
            out["srcs"] = batched_srcs
    except Exception:
        return None

    # hs_last, pred_boxes, pred_scores: stack along batch dim
    for k in ("hs_last", "pred_boxes", "pred_scores"):
        if k not in items[0]:
            continue
        lvl = []
        for it in items:
            t = it.get(k, None)
            if not torch.is_tensor(t):
                return None
            if t.dim() >= 1:
                t = t.unsqueeze(0)
            lvl.append(t)
        v = torch.cat(lvl, dim=0).to(device=device)
        if dtype is not None and v.dtype != dtype:
            v = v.to(dtype=dtype)
        out[k] = v

    return out


def load_gdino_cache_single(
    cache_dir: str,
    *,
    img_path: str,
    target_hw: Tuple[int, int],
    bb_cfg: Optional[dict],
    mode: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[Dict[str, Any]]:
    d = _load_one(cache_dir, img_path=img_path, target_hw=target_hw, bb_cfg=bb_cfg, mode=mode)
    if d is None:
        return None
    out: Dict[str, Any] = {}
    if isinstance(d.get("srcs", None), list):
        srcs = []
        for t in d["srcs"]:
            if not torch.is_tensor(t):
                continue
            if t.dim() == 3:
                t = t.unsqueeze(0)
            t = t.to(device=device)
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype=dtype)
            srcs.append(t)
        out["srcs"] = srcs
    for k in ("hs_last", "pred_boxes", "pred_scores"):
        t = d.get(k, None)
        if torch.is_tensor(t):
            if t.dim() >= 1:
                t = t.unsqueeze(0)
            t = t.to(device=device)
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype=dtype)
            out[k] = t
    return out
