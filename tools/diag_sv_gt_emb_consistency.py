#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS, MODELS


def _q(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    idx = int(round(q * (len(xs) - 1)))
    return float(xs[idx])


def _summ(xs: List[float], qs=(0.1, 0.5, 0.9)) -> Dict[str, float]:
    if not xs:
        return {"n": 0}
    out = {"n": int(len(xs)), "mean": float(np.mean(xs)), "std": float(np.std(xs))}
    for q in qs:
        out[f"p{int(q*100):02d}"] = float(_q(xs, q))
    return out


def _gt_emb_diag_step(
    *,
    gt_inst: torch.Tensor,
    pred_masks: torch.Tensor,
    pred_queries: torch.Tensor,
    frame_i: int,
    state: dict,
    min_iou: float,
    iou_thr: float,
    gt_vis_npoint: int,
    neg_pairs: int,
) -> Dict:
    device = pred_queries.device
    if gt_inst is None or pred_masks is None or pred_queries is None:
        return {"frame": int(frame_i), "skipped": "missing_inputs"}
    if not torch.is_tensor(gt_inst):
        gt_inst = torch.as_tensor(gt_inst)
    gt_inst = gt_inst.to(device=device).long().flatten()
    if gt_inst.numel() == 0:
        return {"frame": int(frame_i), "skipped": "no_gt"}

    if pred_masks.dim() != 2 or pred_queries.dim() != 2:
        return {
            "frame": int(frame_i),
            "skipped": "bad_shapes",
            "pred_masks_shape": list(pred_masks.shape),
            "pred_queries_shape": list(pred_queries.shape),
        }
    if pred_masks.dtype != torch.bool:
        pred_masks = pred_masks.bool()
    n_pred, n_pts = int(pred_masks.shape[0]), int(pred_masks.shape[1])
    if int(pred_queries.shape[0]) != n_pred:
        return {"frame": int(frame_i), "skipped": "shape_mismatch", "n_pred": n_pred, "n_query": int(pred_queries.shape[0])}
    if int(gt_inst.numel()) != n_pts:
        return {"frame": int(frame_i), "skipped": "gt_pred_pts_mismatch", "n_pts_pred": n_pts, "n_pts_gt": int(gt_inst.numel())}

    ids, counts = torch.unique(gt_inst, return_counts=True)
    keep = ids != -1
    ids = ids[keep]
    counts = counts[keep]
    if gt_vis_npoint > 0:
        keep = counts >= int(gt_vis_npoint)
        ids = ids[keep]
        counts = counts[keep]
    if int(ids.numel()) == 0:
        return {"frame": int(frame_i), "skipped": "no_visible_gt"}

    gt_masks = (gt_inst.unsqueeze(0) == ids.unsqueeze(1))  # [n_gt, n_pts]
    gt_sizes = gt_masks.sum(dim=1).float()

    pred_f = pred_masks.float()
    gt_f = gt_masks.float()
    inter = pred_f @ gt_f.t()  # [n_pred, n_gt]
    pred_sz = pred_f.sum(dim=1, keepdim=True)
    union = pred_sz + gt_sizes.unsqueeze(0) - inter
    iou = inter / (union + 1e-6)
    best_iou, best_pi = iou.max(dim=0)  # per-gt

    matched = best_iou >= float(min_iou)
    if int(matched.sum().item()) == 0:
        return {"frame": int(frame_i), "n_gt_vis": int(ids.numel()), "n_gt_matched": 0, "n_pred": n_pred}

    gt_ids_mat = ids[matched]
    pi_mat = best_pi[matched]
    iou_mat = best_iou[matched].detach()

    emb = pred_queries[pi_mat]
    emb = F.normalize(emb.float(), dim=1)

    prev = state.get("prev", {})
    new_prev = prev.copy() if isinstance(prev, dict) else {}
    pos_cos = []
    pos_strong_cos = []
    for k in range(int(gt_ids_mat.numel())):
        gid = int(gt_ids_mat[k].item())
        e_cur = emb[k]
        rec = new_prev.get(gid, None)
        if isinstance(rec, dict):
            prev_f = int(rec.get("frame", -999999))
            e_prev = rec.get("emb", None)
            if prev_f == int(frame_i - 1) and torch.is_tensor(e_prev):
                c = float((e_prev * e_cur).sum().clamp(-1, 1).item())
                pos_cos.append(c)
                if float(iou_mat[k].item()) >= float(iou_thr):
                    pos_strong_cos.append(c)
        new_prev[gid] = {"frame": int(frame_i), "emb": e_cur.detach()}
    state["prev"] = new_prev

    neg_raw = []
    try:
        n = int(emb.shape[0])
        if n >= 2:
            sim = emb @ emb.t()
            mask = ~torch.eye(n, device=device, dtype=torch.bool)
            vals = sim[mask].flatten()
            if vals.numel() > 0 and neg_pairs > 0 and vals.numel() > neg_pairs:
                idx = torch.randperm(vals.numel(), device=device)[:neg_pairs]
                vals = vals[idx]
            neg_raw = [float(x) for x in vals.detach().cpu().tolist()]
    except Exception:
        neg_raw = []

    out = {
        "frame": int(frame_i),
        "n_gt_vis": int(ids.numel()),
        "n_gt_matched": int(matched.sum().item()),
        "n_pred": n_pred,
        "pos_raw": pos_cos,
        "pos_strong_raw": pos_strong_cos,
        "neg_raw": neg_raw,
    }
    # separations
    if pos_cos and neg_raw:
        out["sep_pos_mean_neg_p90"] = float(np.mean(pos_cos) - np.percentile(neg_raw, 90))
    return out


def _parse_scene_and_frame(path: str) -> Tuple[str, Optional[int]]:
    if not path:
        return "unknown", None
    m = re.search(r"(scene\d{4}_\d{2})", path)
    scene_id = m.group(1) if m else "unknown"
    base = os.path.basename(path)
    m2 = re.search(r"(\d+)(?:\.[a-zA-Z0-9]+)?$", base)
    frame_id = int(m2.group(1)) if m2 else None
    return scene_id, frame_id


@dataclass
class SubsetSpec:
    scenes: List[str]
    frames_per_scene: int
    frame_step: int


def build_scene_index(dataset) -> Dict[str, List[Tuple[int, Optional[int]]]]:
    by_scene: Dict[str, List[Tuple[int, Optional[int]]]] = defaultdict(list)
    # Try to use dataset.data_list if available; fallback to get_data_info.
    data_list = getattr(dataset, "data_list", None)
    for idx in range(len(dataset)):
        info = None
        if isinstance(data_list, list) and idx < len(data_list):
            info = data_list[idx]
        if not isinstance(info, dict):
            try:
                info = dataset.get_data_info(idx)
            except Exception:
                info = {}
        lidar_path = info.get("lidar_path", None) or info.get("lidar_points", {}).get("lidar_path", None)
        scene_id, frame_id = _parse_scene_and_frame(str(lidar_path) if lidar_path else "")
        by_scene[scene_id].append((idx, frame_id))
    # Sort by frame_id when available.
    for scene_id, items in by_scene.items():
        items.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else x[0]))
    return by_scene


def choose_subset(
    by_scene: Dict[str, List[Tuple[int, Optional[int]]]],
    num_scenes: int,
    frames_per_scene: int,
    frame_step: int,
    seed: int,
) -> SubsetSpec:
    scenes = [s for s, items in by_scene.items() if s != "unknown" and len(items) >= 2]
    rng = np.random.RandomState(seed)
    rng.shuffle(scenes)
    scenes = scenes[: max(1, int(num_scenes))]
    return SubsetSpec(scenes=scenes, frames_per_scene=int(frames_per_scene), frame_step=int(frame_step))


def iter_indices_for_subset(by_scene, subset: SubsetSpec) -> List[Tuple[str, List[int]]]:
    out = []
    for scene_id in subset.scenes:
        items = by_scene.get(scene_id, [])
        idxs = [i for i, _ in items]
        if subset.frame_step > 1:
            idxs = idxs[:: subset.frame_step]
        if subset.frames_per_scene > 0:
            idxs = idxs[: subset.frames_per_scene]
        if len(idxs) >= 2:
            out.append((scene_id, idxs))
    return out


def load_state_dict_forgiving(model: torch.nn.Module, ckpt_path: Path) -> Dict:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "missing_keys_head": list(missing)[:20],
        "unexpected_keys_head": list(unexpected)[:20],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-scenes", type=int, default=5)
    ap.add_argument("--frames-per-scene", type=int, default=20)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gt-vis-npoint", type=int, default=100)
    ap.add_argument("--min-iou", type=float, default=0.1)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--neg-pairs", type=int, default=2048)
    ap.add_argument("--scenes-json", type=Path, default=None)
    args = ap.parse_args()

    init_default_scope("mmdet3d")
    cfg = Config.fromfile(str(args.config))
    # Ensure we export instance queries for diagnostics; do not change default behavior otherwise.
    cfg.model.test_cfg.export_instance_queries = True
    model = MODELS.build(cfg.model)
    load_info = load_state_dict_forgiving(model, args.checkpoint)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Build dataset from test_dataloader (SV val list).
    ds_cfg = cfg.get("test_dataloader", cfg.get("val_dataloader", None))
    if ds_cfg is None:
        raise RuntimeError("Config missing test_dataloader/val_dataloader.")
    ds_cfg = ds_cfg.dataset
    dataset = DATASETS.build(ds_cfg)
    by_scene = build_scene_index(dataset)

    if args.scenes_json and args.scenes_json.exists():
        scenes = json.loads(args.scenes_json.read_text())
        subset = SubsetSpec(
            scenes=list(scenes),
            frames_per_scene=args.frames_per_scene,
            frame_step=args.frame_step,
        )
    else:
        subset = choose_subset(
            by_scene,
            num_scenes=args.num_scenes,
            frames_per_scene=args.frames_per_scene,
            frame_step=args.frame_step,
            seed=args.seed,
        )

    groups = iter_indices_for_subset(by_scene, subset)
    if not groups:
        raise RuntimeError("No valid scenes with >=2 frames found for subset.")

    pos_all: List[float] = []
    pos_strong_all: List[float] = []
    neg_all: List[float] = []
    sep_all: List[float] = []
    per_scene = {}
    skip_global: Dict[str, int] = defaultdict(int)
    skip_examples: Dict[str, List[dict]] = defaultdict(list)

    diag_cfg = {
        "min_iou": float(args.min_iou),
        "iou_thr": float(args.iou_thr),
        "gt_vis_npoint": int(args.gt_vis_npoint),
        "neg_pairs": int(args.neg_pairs),
    }

    for scene_id, idxs in groups:
        state = {}
        scene_pos: List[float] = []
        scene_pos_strong: List[float] = []
        scene_neg: List[float] = []
        scene_sep: List[float] = []
        scene_skip: Dict[str, int] = defaultdict(int)
        scene_skip_examples: Dict[str, List[dict]] = defaultdict(list)

        for local_fi, ds_idx in enumerate(idxs):
            data = dataset[ds_idx]
            batch = pseudo_collate([data])
            with torch.no_grad():
                outputs = model.test_step(batch)
            sample = outputs[0]

            gt_inst = None
            # Prefer eval_ann_info (works for baseline SV configs where gt_pts_seg may not pack masks).
            try:
                if isinstance(getattr(sample, "eval_ann_info", None), dict):
                    gt_inst = sample.eval_ann_info.get("pts_instance_mask", None)
            except Exception:
                gt_inst = None
            if gt_inst is None:
                gt_inst = getattr(getattr(sample, "gt_pts_seg", None), "pts_instance_mask", None)
            if gt_inst is None:
                scene_skip["missing_gt_inst"] += 1
                skip_global["missing_gt_inst"] += 1
                if len(scene_skip_examples["missing_gt_inst"]) < 3:
                    scene_skip_examples["missing_gt_inst"].append(
                        {"scene": scene_id, "ds_idx": int(ds_idx), "frame": int(local_fi)})
                if len(skip_examples["missing_gt_inst"]) < 10:
                    skip_examples["missing_gt_inst"].append(
                        {"scene": scene_id, "ds_idx": int(ds_idx), "frame": int(local_fi)})
                continue
            pred = sample.pred_pts_seg
            if not hasattr(pred, "instance_queries"):
                raise RuntimeError(
                    "pred_pts_seg missing instance_queries. "
                    "Set model.test_cfg.export_instance_queries=True and ensure code exports queries.")

            pred_masks_np = pred.get("pts_instance_mask", None)
            if isinstance(pred_masks_np, list) and pred_masks_np:
                pred_masks_np = pred_masks_np[0]
            if pred_masks_np is None:
                scene_skip["missing_pred_masks"] += 1
                skip_global["missing_pred_masks"] += 1
                continue
            pred_masks = torch.as_tensor(pred_masks_np, device=device)
            pred_queries = pred.instance_queries.to(device=device)

            out = _gt_emb_diag_step(
                gt_inst=gt_inst,
                pred_masks=pred_masks,
                pred_queries=pred_queries,
                frame_i=int(local_fi),
                state=state,
                min_iou=float(args.min_iou),
                iou_thr=float(args.iou_thr),
                gt_vis_npoint=int(args.gt_vis_npoint),
                neg_pairs=int(args.neg_pairs),
            )
            if not isinstance(out, dict):
                scene_skip["bad_out"] += 1
                skip_global["bad_out"] += 1
                continue
            if out.get("skipped", None):
                reason = str(out.get("skipped"))
                scene_skip[reason] += 1
                skip_global[reason] += 1
                # keep a few examples for debugging
                if len(scene_skip_examples[reason]) < 3:
                    ex = {k: out.get(k) for k in ("frame", "skipped", "n_pred", "n_query", "n_pts_pred", "n_pts_gt", "pred_masks_shape", "pred_queries_shape")}
                    ex.update({"scene": scene_id, "ds_idx": int(ds_idx)})
                    scene_skip_examples[reason].append(ex)
                if len(skip_examples[reason]) < 10:
                    ex = {k: out.get(k) for k in ("frame", "skipped", "n_pred", "n_query", "n_pts_pred", "n_pts_gt", "pred_masks_shape", "pred_queries_shape")}
                    ex.update({"scene": scene_id, "ds_idx": int(ds_idx)})
                    skip_examples[reason].append(ex)
                continue
            xs = out.get("pos_raw", [])
            if isinstance(xs, list):
                scene_pos.extend(xs)
                pos_all.extend(xs)
            xs = out.get("pos_strong_raw", [])
            if isinstance(xs, list):
                scene_pos_strong.extend(xs)
                pos_strong_all.extend(xs)
            xs = out.get("neg_raw", [])
            if isinstance(xs, list):
                scene_neg.extend(xs)
                neg_all.extend(xs)
            if out.get("sep_pos_mean_neg_p90", None) is not None:
                scene_sep.append(float(out["sep_pos_mean_neg_p90"]))
                sep_all.append(float(out["sep_pos_mean_neg_p90"]))

        per_scene[scene_id] = {
            "n_frames": len(idxs),
            "skip_counts": dict(scene_skip),
            "skip_examples": {k: v for k, v in scene_skip_examples.items() if v},
            "pos_p50_list": scene_pos,
            "pos_strong_p50_list": scene_pos_strong,
            "neg_p90_list": scene_neg,
            "sep_list": scene_sep,
            "pos_p50": _summ(scene_pos, qs=(0.1, 0.5, 0.9)),
            "pos_strong_p50": _summ(scene_pos_strong, qs=(0.1, 0.5, 0.9)),
            "neg_p90": _summ(scene_neg, qs=(0.1, 0.5, 0.9)),
            "sep": _summ(scene_sep, qs=(0.1, 0.5, 0.9)),
        }

    out_json = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "load_info": load_info,
        "subset": {
            "num_scenes": len([s for s, _ in groups]),
            "scenes": [s for s, _ in groups],
            "frames_per_scene": args.frames_per_scene,
            "frame_step": args.frame_step,
            "seed": args.seed,
        },
        "diag_cfg": diag_cfg,
        "global": {
            "pos_p50": _summ(pos_all),
            "pos_strong_p50": _summ(pos_strong_all),
            "neg_p90": _summ(neg_all),
            "sep_pos50_neg90": _summ(sep_all),
            "skip_counts": dict(skip_global),
            "skip_examples": {k: v for k, v in skip_examples.items() if v},
        },
        "per_scene": per_scene,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_json, indent=2))
    if not args.scenes_json:
        scenes_path = args.out.with_suffix(".scenes.json")
        scenes_path.write_text(json.dumps(out_json["subset"]["scenes"], indent=2))
        print(f"[saved] scenes: {scenes_path}")
    print(f"[saved] diag: {args.out}")


if __name__ == "__main__":
    main()
