#!/usr/bin/env python3
"""Make a random subset of OneFormer3D-style infos pkl.

Supports two modes:
- entry: sample K entries from data_list (useful when each entry is a scene)
- sv_scene: sample K unique ScanNet SV scenes (when each entry is a frame)

Usage:
  python tools/make_subset_infos.py \
    --in-pkl data/scannet200-mv_fast/scannet200_mv_oneformer3d_infos_val.pkl \
    --out-pkl data/scannet200-mv_fast/subsets/scannet200_mv_infos_val_subset64_seed0.pkl \
    --num-scenes 64 \
    --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from typing import Any, Dict, List


def _load_infos(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and isinstance(obj.get("data_list", None), list):
        return obj
    if isinstance(obj, list):
        return {"data_list": obj}
    raise ValueError(f"Unsupported infos pkl format: {type(obj)} keys={getattr(obj, 'keys', lambda: [])()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-pkl", required=True)
    ap.add_argument("--out-pkl", required=True)
    ap.add_argument("--num-scenes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--mode",
        type=str,
        default="entry",
        choices=["entry", "sv_scene"],
        help="entry: sample K entries; sv_scene: sample K unique SV scenes (by img_path prefix).",
    )
    ap.add_argument(
        "--max-entries-per-scene",
        type=int,
        default=0,
        help="Only for --mode sv_scene. 0 means keep all entries for each chosen scene.",
    )
    args = ap.parse_args()

    infos = _load_infos(args.in_pkl)
    data_list: List[Dict[str, Any]] = infos["data_list"]
    n = len(data_list)
    rng = random.Random(int(args.seed))
    k = int(args.num_scenes)
    if k <= 0:
        raise ValueError(f"--num-scenes must be >=1, got {k}")

    chosen = []
    subset: List[Dict[str, Any]] = []
    subset_meta: Dict[str, Any] = {
        "in_pkl": os.path.abspath(args.in_pkl),
        "num_scenes": k,
        "seed": int(args.seed),
        "mode": str(args.mode),
    }

    if args.mode == "entry":
        if k > n:
            raise ValueError(f"--num-scenes must be in [1,{n}] for mode=entry, got {k}")
        idxs = list(range(n))
        rng.shuffle(idxs)
        chosen = sorted(idxs[:k])
        subset = [data_list[i] for i in chosen]
        subset_meta["chosen_indices"] = chosen
    else:
        # SV pkl is frame-level. Group by ScanNet scene name extracted from img_path:
        #   2D/scene0613_00/color/3200.jpg -> scene0613_00
        def _scene_from_img_path(p: str) -> str:
            if not isinstance(p, str) or not p:
                return ""
            parts = p.replace("\\", "/").split("/")
            # expected: ["2D", "sceneXXXX_YY", "color", ...]
            if len(parts) >= 2 and parts[1].startswith("scene"):
                return parts[1]
            # fallback: find first token like "scene????_??"
            for token in parts:
                if token.startswith("scene") and "_" in token:
                    return token
            return ""

        scene_to_indices: Dict[str, List[int]] = {}
        for i, info in enumerate(data_list):
            scene = _scene_from_img_path(info.get("img_path", ""))
            if not scene:
                continue
            scene_to_indices.setdefault(scene, []).append(i)

        scenes = sorted(scene_to_indices.keys())
        if k > len(scenes):
            raise ValueError(f"--num-scenes={k} > available scenes={len(scenes)} in SV pkl")

        scenes_shuf = scenes[:]
        rng.shuffle(scenes_shuf)
        chosen_scenes = sorted(scenes_shuf[:k])
        subset_meta["chosen_scenes"] = chosen_scenes

        max_per = int(args.max_entries_per_scene)
        for scene in chosen_scenes:
            idxs = scene_to_indices[scene][:]
            rng.shuffle(idxs)
            if max_per > 0:
                idxs = idxs[:max_per]
            chosen.extend(sorted(idxs))

        chosen = sorted(set(chosen))
        subset = [data_list[i] for i in chosen]
        subset_meta["chosen_indices"] = chosen
        subset_meta["max_entries_per_scene"] = int(args.max_entries_per_scene)

    out = dict(infos)
    out["data_list"] = subset
    out["subset_meta"] = subset_meta

    out_dir = os.path.dirname(args.out_pkl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(out, f)

    with open(args.out_pkl + ".json", "w", encoding="utf-8") as f:
        json.dump(out["subset_meta"], f, ensure_ascii=False, indent=2)

    print(f"Wrote subset pkl: {args.out_pkl} ({k} mode={args.mode}, entries={len(subset)})")


if __name__ == "__main__":
    main()
