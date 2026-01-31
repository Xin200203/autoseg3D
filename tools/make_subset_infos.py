#!/usr/bin/env python3
"""
Make a small subset infos .pkl for quick smoke tests.

This is intentionally lightweight (pickle-in/pickle-out) and does not require
torch/mmengine. It works for both SV and MV ScanNet200-style infos:
- SV items often contain:  img_path (str)
- MV items often contain:  img_paths (list[str])

Example:
  python tools/make_subset_infos.py \
    --in /path/to/scannet200_sv_oneformer3d_infos_train.pkl \
    --out /path/to/subset.pkl \
    --scene scene0382_00 \
    --max-items 4
"""

import argparse
import json
import os
import os.path as osp
import pickle
import re
from typing import Any, Dict, List, Optional, Sequence


def _is_numpy_obj(x: Any) -> bool:
    mod = getattr(getattr(x, "__class__", None), "__module__", "")
    return isinstance(mod, str) and mod.startswith("numpy")


def _sanitize(obj: Any) -> Any:
    """Convert numpy-containing structures into pure-Python types.

    This makes the output pickle robust across numpy major versions (e.g.
    numpy>=2 pickles may not unpickle under numpy<2 due to internal module
    paths like `numpy._core`).
    """
    if _is_numpy_obj(obj):
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass
        if hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:
                pass
        return str(obj)

    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize(v) for v in obj)
    return obj


def _extract_rel_img_paths(item: Dict[str, Any]) -> List[str]:
    if "img_path" in item and isinstance(item["img_path"], str):
        return [item["img_path"]]
    if "img_paths" in item and isinstance(item["img_paths"], list):
        return [p for p in item["img_paths"] if isinstance(p, str)]
    return []


def _match_scene(item: Dict[str, Any], *, scene: Optional[str], scene_re: Optional[re.Pattern]) -> bool:
    if scene is None and scene_re is None:
        return True
    paths = _extract_rel_img_paths(item)
    if not paths:
        return False
    # Most ScanNet paths embed the scene id: "2D/sceneXXXX_XX/color/....jpg"
    s = " ".join(paths)
    if scene is not None and scene not in s:
        return False
    if scene_re is not None and scene_re.search(s) is None:
        return False
    return True


def _load_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "data_list" not in obj:
        raise ValueError(f"unexpected pkl format: {path}")
    if not isinstance(obj["data_list"], list):
        raise ValueError(f"unexpected data_list type: {type(obj['data_list']).__name__}")
    return obj


def _save_pkl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="input infos .pkl")
    ap.add_argument("--out", dest="out_path", required=True, help="output subset infos .pkl")
    ap.add_argument("--scene", default=None, help="scene id substring, e.g. scene0382_00")
    ap.add_argument("--scene-regex", default=None, help="regex applied on image paths")
    ap.add_argument("--max-items", type=int, default=1, help="max data_list items to keep")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="truncate MV lists (img_paths/poses/pts_paths/...) to first N frames; <=0 disables",
    )
    ap.add_argument("--seed", type=int, default=0, help="reserved for future random sampling")
    args = ap.parse_args()

    in_path = osp.abspath(osp.expanduser(args.in_path))
    out_path = osp.abspath(osp.expanduser(args.out_path))

    scene_re = re.compile(args.scene_regex) if args.scene_regex else None
    base = _load_pkl(in_path)
    data_list: Sequence[Dict[str, Any]] = base["data_list"]

    kept: List[Dict[str, Any]] = []
    for it in data_list:
        if not isinstance(it, dict):
            continue
        if not _match_scene(it, scene=args.scene, scene_re=scene_re):
            continue
        # Optionally truncate MV per-frame lists to make a tiny sample.
        if int(args.max_frames) > 0 and isinstance(it.get("img_paths", None), list):
            img_paths = [p for p in it.get("img_paths", []) if isinstance(p, str)]
            n = min(int(args.max_frames), len(img_paths))
            if n > 0:
                it = dict(it)  # shallow copy
                it["img_paths"] = img_paths[:n]
                # Common per-frame parallel lists in MV infos.
                for k in (
                    "poses",
                    "pts_paths",
                    "super_pts_paths",
                    "pts_instance_mask_paths",
                    "pts_semantic_mask_paths",
                ):
                    v = it.get(k, None)
                    if isinstance(v, list) and len(v) >= n:
                        it[k] = v[:n]
        kept.append(it)
        if len(kept) >= int(args.max_items):
            break

    subset_meta = {
        "source": in_path,
        "scene": args.scene,
        "scene_regex": args.scene_regex,
        "max_items": int(args.max_items),
        "max_frames": int(args.max_frames),
        "seed": int(args.seed),
        "kept": int(len(kept)),
    }

    out = dict(base)
    out["data_list"] = kept
    out["subset_meta"] = subset_meta
    _save_pkl(out_path, _sanitize(out))

    # Sidecar JSON for quick inspection
    try:
        with open(out_path + ".json", "w", encoding="utf-8") as f:
            json.dump(subset_meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(f"[subset] in={in_path}")
    print(f"[subset] out={out_path}")
    print(f"[subset] kept={len(kept)}")
    if kept:
        print(f"[subset] first_img_paths={_extract_rel_img_paths(kept[0])[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
