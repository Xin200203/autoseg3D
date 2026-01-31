#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean


def _q(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    idx = int(round(q * (len(xs) - 1)))
    return xs[idx]


def _summ(xs):
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": float(mean(xs)),
        "p10": float(_q(xs, 0.10)),
        "p50": float(_q(xs, 0.50)),
        "p90": float(_q(xs, 0.90)),
    }


def load_online_monitor_json(path: Path):
    data = json.loads(path.read_text())
    # online_monitor.json may be:
    #   - list[scene_dict] (common when evaluator dumps per-scene entries)
    #   - dict with "frames" (single-scene debug)
    if isinstance(data, list):
        scenes = [x for x in data if isinstance(x, dict)]
        frames = []
        for sc in scenes:
            _fs = sc.get("frames", [])
            if isinstance(_fs, list):
                frames.extend([f for f in _fs if isinstance(f, dict)])
    elif isinstance(data, dict):
        frames = data.get("frames", [])
        if not isinstance(frames, list):
            frames = []
    else:
        frames = []

    pos_raw_all = []
    neg_raw_all = []
    pos_mean_by_frame = []
    neg_mean_by_frame = []
    missing = 0
    assoc_missing = 0
    for fr in frames:
        assoc = fr.get("assoc", None)
        if not isinstance(assoc, dict):
            assoc_missing += 1
            continue
        mc = assoc.get("match_cos", None)
        if not isinstance(mc, dict):
            missing += 1
            continue
        if isinstance(mc.get("pos_raw", None), list):
            pos_raw_all.extend([float(x) for x in mc["pos_raw"] if x is not None])
        if isinstance(mc.get("neg_raw", None), list):
            neg_raw_all.extend([float(x) for x in mc["neg_raw"] if x is not None])
        pos = mc.get("pos", {})
        neg = mc.get("neg", {})
        if isinstance(pos, dict) and "mean" in pos:
            pos_mean_by_frame.append(float(pos["mean"]))
        if isinstance(neg, dict) and "mean" in neg:
            neg_mean_by_frame.append(float(neg["mean"]))
    return {
        "frames_total": len(frames),
        "frames_with_match_cos": len(pos_mean_by_frame),
        "frames_missing_match_cos": missing,
        "frames_missing_assoc": assoc_missing,
        "pos_raw": pos_raw_all,
        "neg_raw": neg_raw_all,
        "pos_mean_by_frame": pos_mean_by_frame,
        "neg_mean_by_frame": neg_mean_by_frame,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("online_monitor_json", type=Path)
    args = ap.parse_args()
    d = load_online_monitor_json(args.online_monitor_json)

    print(f"online_monitor_json: {args.online_monitor_json}")
    print(f"frames_total: {d['frames_total']}")
    print(f"frames_missing_assoc: {d['frames_missing_assoc']}")
    print(f"frames_with_match_cos: {d['frames_with_match_cos']}")
    print(f"frames_missing_match_cos: {d['frames_missing_match_cos']}")

    if d["pos_raw"] and d["neg_raw"]:
        print("pos_raw:", _summ(d["pos_raw"]))
        print("neg_raw:", _summ(d["neg_raw"]))
        print("separation (pos_mean - neg_mean):", float(mean(d["pos_raw"]) - mean(d["neg_raw"])))
    else:
        print("pos_raw: (missing)  -> enable model.test_cfg.online_monitor.bbox_center_diag.match_cos.save_raw=True")
        print("neg_raw: (missing)  -> enable model.test_cfg.online_monitor.bbox_center_diag.match_cos.save_raw=True")
        if d["pos_mean_by_frame"]:
            print("pos_mean_by_frame:", _summ(d["pos_mean_by_frame"]))
        if d["neg_mean_by_frame"]:
            print("neg_mean_by_frame:", _summ(d["neg_mean_by_frame"]))


if __name__ == "__main__":
    main()
