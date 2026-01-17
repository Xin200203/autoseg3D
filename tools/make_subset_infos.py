#!/usr/bin/env python3
"""Make a random subset of OneFormer3D-style infos pkl (scene-level entries).

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
    args = ap.parse_args()

    infos = _load_infos(args.in_pkl)
    data_list: List[Dict[str, Any]] = infos["data_list"]
    n = len(data_list)
    k = int(args.num_scenes)
    if k <= 0 or k > n:
        raise ValueError(f"--num-scenes must be in [1,{n}], got {k}")

    rng = random.Random(int(args.seed))
    idxs = list(range(n))
    rng.shuffle(idxs)
    chosen = sorted(idxs[:k])
    subset = [data_list[i] for i in chosen]

    out = dict(infos)
    out["data_list"] = subset
    out["subset_meta"] = {
        "in_pkl": os.path.abspath(args.in_pkl),
        "num_scenes": k,
        "seed": int(args.seed),
        "chosen_indices": chosen,
    }

    out_dir = os.path.dirname(args.out_pkl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(out, f)

    with open(args.out_pkl + ".json", "w", encoding="utf-8") as f:
        json.dump(out["subset_meta"], f, ensure_ascii=False, indent=2)

    print(f"Wrote subset pkl: {args.out_pkl} ({k}/{n} scenes)")


if __name__ == "__main__":
    main()

