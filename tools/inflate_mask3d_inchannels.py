#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def inflate_conv0_kernel(state_dict: dict, new_in_channels: int) -> None:
    key = "backbone.conv0p1s1.kernel"
    if key not in state_dict:
        raise KeyError(f"Missing key in checkpoint: {key}")
    w = state_dict[key]
    if w.ndim != 3:
        raise ValueError(f"Unexpected conv0 kernel shape: {tuple(w.shape)} (expected 3D)")
    out_c, in_c, kv = w.shape
    if int(new_in_channels) == int(in_c):
        return
    if new_in_channels < in_c:
        state_dict[key] = w[:, :new_in_channels, :].contiguous()
        return
    w_new = w.new_zeros((out_c, new_in_channels, kv))
    w_new[:, :in_c, :] = w
    state_dict[key] = w_new.contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source checkpoint path (mask3d_scannet200.pth)")
    ap.add_argument("--dst", required=True, help="Destination checkpoint path")
    ap.add_argument("--in-channels", type=int, required=True, help="New input channels for backbone conv0")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(src), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state_dict is not a dict")

    inflate_conv0_kernel(state, int(args.in_channels))
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt["state_dict"] = state
        torch.save(ckpt, str(dst))
    else:
        torch.save(state, str(dst))

    print(f"[inflate] saved: {dst}")


if __name__ == "__main__":
    main()

