"""
Sparse FPN builder for point-wise 2D features (MinkowskiEngine).

This is copied/adapted from the ESAM/OneFormer3D DINO-FPN implementation
to keep the exact same semantics:
  - take per-point features + (batched) sparse coords at stride=1
  - build a 3D pyramid [s1, s2, s4, s8, s16] via sparse max-pooling

AutoSeg3D usage:
  - sample point-wise GDINO srcs (image-level feature maps)
  - quantize with the same coord source as backbone (elastic or xyz_aug)
  - build_sparse_fpn(coords, feats) -> pass to Res16UNet34C(..., dino_feats=fpn)
"""

from __future__ import annotations

from typing import List, Optional

import MinkowskiEngine as ME
import torch


def build_sparse_fpn(
    coords: torch.Tensor,
    feats: torch.Tensor,
    tensor_stride: int = 1,
    dimension: int = 3,
    coordinate_manager: Optional[ME.CoordinateManager] = None,
    pool: Optional[ME.MinkowskiMaxPooling] = None,
) -> List[ME.SparseTensor]:
    """Build sparse FPN pyramid from coords+feats.

    Args:
        coords: (N, D+1) int32 coords with batch index in the first column.
        feats:  (N, C) float features (e.g., point-wise GDINO features).
        tensor_stride: input stride (typically 1).
        dimension: spatial dimension (default 3).
        coordinate_manager: optional shared CoordinateManager.
        pool: optional pooling op. Defaults to maxpool(k=2,s=2).

    Returns:
        [s1, s2, s4, s8, s16] sparse tensors.
    """
    if pool is None:
        pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=dimension)

    feats = feats.float()
    coords = coords.to(dtype=torch.int32)

    x_s1 = ME.SparseTensor(
        features=feats,
        coordinates=coords,
        tensor_stride=tensor_stride,
        coordinate_manager=coordinate_manager,
    )
    x_s2 = pool(x_s1)
    x_s4 = pool(x_s2)
    x_s8 = pool(x_s4)
    x_s16 = pool(x_s8)
    return [x_s1, x_s2, x_s4, x_s8, x_s16]

