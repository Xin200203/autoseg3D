#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import pickle
import re
import sys
from typing import Iterable, List, Tuple

import numpy as np
import torch
from mmengine.config import Config
from PIL import Image

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from oneformer3d.gdino_backbone import GroundingDINOBackbone  # noqa: E402
from oneformer3d.gdino_cache import gdino_cache_path, save_gdino_cache  # noqa: E402


def _find_resize_target(pipeline) -> Tuple[int, int]:
    try:
        for t in pipeline:
            if isinstance(t, dict) and t.get("type", "") == "ResizeForGDINO":
                hw = t.get("target_size", (420, 560))
                return int(hw[0]), int(hw[1])
    except Exception:
        pass
    return 420, 560


def _iter_img_paths_from_infos(infos_obj, *, data_root: str) -> Iterable[str]:
    if isinstance(infos_obj, dict):
        data_list = infos_obj.get("data_list", None)
        if isinstance(data_list, list):
            for it in data_list:
                if not isinstance(it, dict):
                    continue
                if "img_path" in it and isinstance(it["img_path"], str):
                    yield osp.join(data_root, it["img_path"])
                elif "img_paths" in it and isinstance(it["img_paths"], list):
                    for p in it["img_paths"]:
                        if isinstance(p, str):
                            yield osp.join(data_root, p)
        return
    if isinstance(infos_obj, list):
        for it in infos_obj:
            if not isinstance(it, dict):
                continue
            if "img_path" in it and isinstance(it["img_path"], str):
                yield osp.join(data_root, it["img_path"])
            elif "img_paths" in it and isinstance(it["img_paths"], list):
                for p in it["img_paths"]:
                    if isinstance(p, str):
                        yield osp.join(data_root, p)


def _open_and_resize(img_path: str, *, target_hw: Tuple[int, int]) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    h, w = int(target_hw[0]), int(target_hw[1])
    if img.size != (w, h):
        img = img.resize((w, h), resample=Image.BILINEAR)
    return img


def _pil_to_tensor(img: Image.Image, *, device: torch.device) -> torch.Tensor:
    arr = np.asarray(img).copy()
    t = torch.from_numpy(arr).to(device=device).float() / 255.0
    if t.dim() == 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1).contiguous()
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="MMEngine config (.py)")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument(
        "--ann-file",
        default=None,
        help="override dataset.ann_file (path relative to data_root or absolute path)",
    )
    ap.add_argument(
        "--data-root",
        default=None,
        help="override dataset.data_root (absolute or relative to AutoSeg3D repo root)",
    )
    ap.add_argument(
        "--scene",
        default=None,
        help="only process images whose relative path contains this scene id (e.g. scene0382_00)",
    )
    ap.add_argument(
        "--scene-regex",
        default=None,
        help="only process images whose relative path matches this regex (applied to rel path like '2D/sceneXXXX_XX/...')",
    )
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--mode", choices=["full", "backbone"], default="full")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing cache files")
    ap.add_argument("--device", default=None, help="override GDINO device, e.g. cuda:0")
    args = ap.parse_args()

    repo_root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    cfg = Config.fromfile(args.config)
    dl = getattr(cfg, f"{args.split}_dataloader", None)
    if dl is None:
        raise SystemExit(f"missing {args.split}_dataloader in config")
    dataset = dl.get("dataset", None) if isinstance(dl, dict) else None
    if not isinstance(dataset, dict):
        raise SystemExit(f"{args.split}_dataloader.dataset must be a dict")
    data_root = args.data_root or dataset.get("data_root", None)
    ann_file = args.ann_file or dataset.get("ann_file", None)
    pipeline = dataset.get("pipeline", None)
    if not isinstance(data_root, str) or not isinstance(ann_file, str):
        raise SystemExit("dataset.data_root / dataset.ann_file missing")
    if not isinstance(pipeline, list):
        pipeline = []

    if not osp.isabs(data_root):
        data_root = osp.join(repo_root, data_root)

    ann_path = ann_file
    if not osp.isabs(ann_path):
        ann_path = osp.join(data_root, ann_path)
    if not osp.exists(ann_path):
        raise SystemExit(f"ann_file not found: {ann_path}")

    target_hw = _find_resize_target(pipeline)
    model_cfg = getattr(cfg, "model", None)
    if not isinstance(model_cfg, dict):
        raise SystemExit("config.model must be a dict")
    bb_cfg = model_cfg.get("gdino_backbone", None)
    if not isinstance(bb_cfg, dict):
        raise SystemExit("config.model.gdino_backbone missing or not a dict")

    repo_dir = bb_cfg.get("repo_dir", None)
    config_path = bb_cfg.get("config_path", None)
    checkpoint = bb_cfg.get("checkpoint", None)
    caption = bb_cfg.get("caption", "object.")
    device = args.device or bb_cfg.get("device", "cuda")
    if not (isinstance(repo_dir, str) and isinstance(config_path, str) and isinstance(checkpoint, str)):
        raise SystemExit("gdino_backbone requires repo_dir/config_path/checkpoint")

    save_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print(f"[GDINO cache] ann={ann_path}")
    print(f"[GDINO cache] target_hw={target_hw} mode={args.mode} dtype={args.dtype} batch_size={args.batch_size}")
    print(f"[GDINO cache] out_dir={osp.abspath(args.cache_dir)}")

    gdino = GroundingDINOBackbone(
        repo_dir=repo_dir,
        config_path=config_path,
        checkpoint=checkpoint,
        device=str(device),
        caption=str(caption),
    )

    with open(ann_path, "rb") as f:
        infos_obj = pickle.load(f)

    scene_sub = str(args.scene) if args.scene else None
    scene_re = re.compile(args.scene_regex) if args.scene_regex else None

    # Build unique, existing image list.
    seen = set()
    img_paths: List[str] = []
    for p in _iter_img_paths_from_infos(infos_obj, data_root=data_root):
        # Filter by scene on the *relative* path (stable across machines).
        rel = p
        if osp.isabs(p):
            try:
                rel = osp.relpath(p, data_root)
            except Exception:
                rel = p
        if scene_sub is not None and scene_sub not in rel:
            continue
        if scene_re is not None and scene_re.search(rel) is None:
            continue

        apath = osp.abspath(p)
        if apath in seen:
            continue
        seen.add(apath)
        if osp.exists(apath):
            img_paths.append(apath)

    if args.limit and args.limit > 0:
        img_paths = img_paths[: int(args.limit)]

    total = len(img_paths)
    if total == 0:
        print("[GDINO cache] no images found")
        return 0

    def _chunks(xs: List[str], n: int):
        n = max(int(n), 1)
        for i in range(0, len(xs), n):
            yield xs[i : i + n]

    done = 0
    for batch in _chunks(img_paths, args.batch_size):
        # Skip-existing check (fast): if all exist and not overwrite.
        cache_paths = [
            gdino_cache_path(
                args.cache_dir, img_path=p, target_hw=target_hw, bb_cfg=bb_cfg, mode=args.mode
            )
            for p in batch
        ]
        if (not args.overwrite) and all(osp.exists(cp) for cp in cache_paths):
            done += len(batch)
            if done % 100 == 0 or done == total:
                print(f"[GDINO cache] {done}/{total} (skipped existing)")
            continue

        imgs = []
        keep_paths = []
        keep_cache_paths = []
        for p, cp in zip(batch, cache_paths):
            if (not args.overwrite) and osp.exists(cp):
                done += 1
                continue
            try:
                imgs.append(_open_and_resize(p, target_hw=target_hw))
                keep_paths.append(p)
                keep_cache_paths.append(cp)
            except Exception as e:
                print(f"[GDINO cache][warn] skip img_load_failed: {p} err={repr(e)}")
                done += 1

        if not keep_paths:
            continue

        img_t = torch.stack([_pil_to_tensor(im, device=gdino.device) for im in imgs], dim=0)
        with torch.no_grad():
            out = gdino(img_t, backbone_only=(args.mode == "backbone"))

        # Split per-sample and save.
        srcs = out.get("srcs", None)
        hs_last = out.get("hs_last", None)
        pred_boxes = out.get("pred_boxes", None)
        pred_scores = out.get("pred_scores", None)
        for i, (p, cp) in enumerate(zip(keep_paths, keep_cache_paths)):
            per_out = {}
            if isinstance(srcs, list):
                per_out["srcs"] = [t[i : i + 1] for t in srcs if torch.is_tensor(t)]
            if torch.is_tensor(hs_last):
                per_out["hs_last"] = hs_last[i : i + 1]
            if torch.is_tensor(pred_boxes):
                per_out["pred_boxes"] = pred_boxes[i : i + 1]
            if torch.is_tensor(pred_scores):
                per_out["pred_scores"] = pred_scores[i : i + 1]

            try:
                save_gdino_cache(
                    cp,
                    img_path=p,
                    target_hw=target_hw,
                    bb_cfg=bb_cfg,
                    mode=args.mode,
                    out=per_out,
                    dtype=save_dtype,
                )
            except Exception as e:
                print(f"[GDINO cache][warn] save_failed: {cp} err={repr(e)}")
            done += 1

        if done % 50 == 0 or done == total:
            print(f"[GDINO cache] {done}/{total}")

    print("[GDINO cache] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
