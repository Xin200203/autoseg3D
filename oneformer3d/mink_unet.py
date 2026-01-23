# Adapted from JonasSchult/Mask3D.
from enum import Enum
from collections.abc import Sequence
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine import MinkowskiReLU

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


class NormType(Enum):
    BATCH_NORM = 0
    INSTANCE_NORM = 1
    INSTANCE_BATCH_NORM = 2


def get_norm(norm_type, n_channels, D, bn_momentum=0.1):
    if norm_type == NormType.BATCH_NORM:
        return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
    elif norm_type == NormType.INSTANCE_NORM:
        return ME.MinkowskiInstanceNorm(n_channels)
    elif norm_type == NormType.INSTANCE_BATCH_NORM:
        return nn.Sequential(
            ME.MinkowskiInstanceNorm(n_channels),
            ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum))
    else:
        raise ValueError(f"Norm type: {norm_type} not supported")


class ConvType(Enum):
    """
    Define the kernel region type
    """

    HYPERCUBE = 0, "HYPERCUBE"
    SPATIAL_HYPERCUBE = 1, "SPATIAL_HYPERCUBE"
    SPATIO_TEMPORAL_HYPERCUBE = 2, "SPATIO_TEMPORAL_HYPERCUBE"
    HYPERCROSS = 3, "HYPERCROSS"
    SPATIAL_HYPERCROSS = 4, "SPATIAL_HYPERCROSS"
    SPATIO_TEMPORAL_HYPERCROSS = 5, "SPATIO_TEMPORAL_HYPERCROSS"
    SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS = (
        6,
        "SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS")

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


# Convert the ConvType var to a RegionType var
conv_to_region_type = {
    # kernel_size = [k, k, k, 1]
    ConvType.HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIO_TEMPORAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIO_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CUBE
}

# int_to_region_type = {m.value: m for m in ME.RegionType}
int_to_region_type = {m: ME.RegionType(m) for m in range(3)}


def convert_region_type(region_type):
    """Convert the integer region_type to the corresponding
    RegionType enum object.
    """
    return int_to_region_type[region_type]


def convert_conv_type(conv_type, kernel_size, D):
    assert isinstance(conv_type, ConvType), "conv_type must be of ConvType"
    region_type = conv_to_region_type[conv_type]
    axis_types = None
    if conv_type == ConvType.SPATIAL_HYPERCUBE:
        # No temporal convolution
        if isinstance(kernel_size, Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCUBE:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.HYPERCUBE:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIAL_HYPERCROSS:
        if isinstance(kernel_size, Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.HYPERCROSS:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCROSS:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS:
        # Define the CUBIC conv kernel for spatial dims
        # and CROSS conv for temp dim
        axis_types = [
            ME.RegionType.HYPER_CUBE,
        ] * 3
        if D == 4:
            axis_types.append(ME.RegionType.HYPER_CROSS)
    return region_type, axis_types, kernel_size


def conv(in_planes,
         out_planes,
         kernel_size,
         stride=1,
         dilation=1,
         bias=False,
         conv_type=ConvType.HYPERCUBE,
         D=-1):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(
        conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=None,  # axis_types JONAS
        dimension=D)

    return ME.MinkowskiConvolution(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D)


def conv_tr(in_planes,
            out_planes,
            kernel_size,
            upsample_stride=1,
            dilation=1,
            bias=False,
            conv_type=ConvType.HYPERCUBE,
            D=-1):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(
        conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        upsample_stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D)

    return ME.MinkowskiConvolutionTranspose(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=upsample_stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D)


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 conv_type=ConvType.HYPERCUBE,
                 bn_momentum=0.1,
                 D=3):
        super().__init__()

        self.conv1 = conv(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            conv_type=conv_type,
            D=D)
        self.norm1 = get_norm(
            self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv2 = conv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            conv_type=conv_type,
            D=D)
        self.norm2 = get_norm(
            self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.relu = MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(BasicBlockBase):
    NORM_TYPE = NormType.BATCH_NORM


class Res16UNetBase(BaseModule):
    """Base class for Minkowski U-Net.

    Args:
        in_channels (int): Number of input channels.
        out_channles (int): Number of output channels.
        config (dict): Extra parameters including
            `dilations`, `conv1_kernel_size`, `bn_momentum`.
        D (int): Conv dimension.
    """
    BLOCK = None
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    def __init__(self,
                 in_channels,
                 out_channels,
                 config,
                 D=3,
                 dino_dim=None,  # Optional: point-wise 2D feature dim for sparse-FPN injection
                 **kwargs):
        self.D = D
        self.dino_dim = dino_dim
        super().__init__()
        self.network_initialization(in_channels, out_channels, config, D)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    norm_type=NormType.BATCH_NORM,
                    bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    D=self.D),
                get_norm(
                    norm_type,
                    planes * block.expansion,
                    D=self.D,
                    bn_momentum=bn_momentum))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_type=self.CONV_TYPE,
                D=self.D))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    conv_type=self.CONV_TYPE,
                    D=self.D))

        return nn.Sequential(*layers)

    def network_initialization(self, in_channels, out_channels, config, D):
        # Setup net_metadata
        dilations = self.DILATIONS
        bn_momentum = config.bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)

        self.bn0 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

        self.conv1p1s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn1 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv2p2s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn2 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv3p4s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn3 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv4p8s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn4 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr4p16s2 = conv_tr(
            self.inplanes,
            self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr4 = get_norm(
            self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            dilation=dilations[4],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr5p8s2 = conv_tr(
            self.inplanes,
            self.PLANES[5],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr5 = get_norm(
            self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            dilation=dilations[5],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr6p4s2 = conv_tr(
            self.inplanes,
            self.PLANES[6],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr6 = get_norm(
            self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(
            self.BLOCK,
            self.PLANES[6],
            self.LAYERS[6],
            dilation=dilations[6],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr7p2s2 = conv_tr(
            self.inplanes,
            self.PLANES[7],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr7 = get_norm(
            self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(
            self.BLOCK,
            self.PLANES[7],
            self.LAYERS[7],
            dilation=dilations[7],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        # self.final = conv(
        #     self.PLANES[7],
        #     out_channels,
        #     kernel_size=1,
        #     stride=1,
        #     bias=True,
        #     D=D)
        self.relu = MinkowskiReLU(inplace=True)

        # Optional: DINO/GDINO sparse-FPN injection (decoder-level fusion)
        # Enabled only when `dino_dim` is provided in config.
        self.use_dino = self.dino_dim is not None
        if self.use_dino:
            self._last_dino_hit = {}
            self._last_dino_fuse = {}
            self._dino_miss_warn_count = 0
            self._dino_min_hit_ratio = float(getattr(config, 'dino_min_hit_ratio', 0.05))
            self._dino_strict = bool(getattr(config, 'dino_strict', False))
            self._dino_residual = bool(getattr(config, 'dino_residual', True))

            def _fuse(in_ch: int, out_ch: int):
                return nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                        dimension=D,
                    ),
                    ME.MinkowskiBatchNorm(out_ch, momentum=bn_momentum),
                    ME.MinkowskiGELU(),
                )

            # level stride=8/4/2/1 (decode stages)
            self.dino_fuse_8 = _fuse(
                self.PLANES[4] + self.PLANES[2] + self.dino_dim,
                self.PLANES[4] + self.PLANES[2],
            )
            self.dino_fuse_4 = _fuse(
                self.PLANES[5] + self.PLANES[1] + self.dino_dim,
                self.PLANES[5] + self.PLANES[1],
            )
            self.dino_fuse_2 = _fuse(
                self.PLANES[6] + self.PLANES[0] + self.dino_dim,
                self.PLANES[6] + self.PLANES[0],
            )
            self.dino_fuse_1 = _fuse(
                self.PLANES[7] + self.INIT_DIM + self.dino_dim,
                self.PLANES[7] + self.INIT_DIM,
            )

            # Zero-init 1x1 conv so residual injection doesn't perturb baseline at start.
            if self._dino_residual:
                for m in (self.dino_fuse_8, self.dino_fuse_4, self.dino_fuse_2, self.dino_fuse_1):
                    try:
                        conv1x1 = m[0]
                        if hasattr(conv1x1, 'kernel') and conv1x1.kernel is not None:
                            conv1x1.kernel.data.zero_()
                    except Exception:
                        pass

    def _fuse_with_dino(self, out_up, out_skip, dino_feat, fuse):
        base_cat = me.cat(out_up, out_skip)
        if dino_feat is None:
            if getattr(self, '_dino_strict', False):
                raise RuntimeError(
                    f"DINO strict: missing dino_feat for tensor_stride={tuple(out_up.tensor_stride)}")
            return base_cat

        try:
            dino_on_out = dino_feat.features_at_coordinates(out_up.coordinates.float())
        except Exception as e:
            if getattr(self, '_dino_strict', False):
                raise RuntimeError(
                    f"DINO strict: features_at_coordinates failed for tensor_stride={tuple(out_up.tensor_stride)}: {repr(e)}")
            return base_cat

        if dino_on_out.dim() != 2 or dino_on_out.shape[0] != out_up.features.shape[0]:
            if getattr(self, '_dino_strict', False):
                raise RuntimeError(
                    "DINO strict: features_at_coordinates returned invalid shape "
                    f"got={tuple(getattr(dino_on_out, 'shape', []))}, expected=({out_up.features.shape[0]}, C)")
            return base_cat

        try:
            hit = (dino_on_out.abs().sum(dim=1) > 1e-8).float().mean().item()
            s = int(getattr(out_up, 'tensor_stride', (0,))[0])
            key = {1: 's1', 2: 's2', 4: 's4', 8: 's8', 16: 's16'}.get(s, f's{s}')
            if hasattr(self, '_last_dino_hit') and isinstance(self._last_dino_hit, dict):
                self._last_dino_hit[key] = float(hit)
            if hit < float(getattr(self, '_dino_min_hit_ratio', 0.0)):
                if getattr(self, '_dino_strict', False):
                    raise RuntimeError(
                        f"DINO strict: low hit ratio={hit:.4f} < {self._dino_min_hit_ratio:.4f} at {key}")
                return base_cat
        except Exception:
            pass

        dino_on_out = ME.SparseTensor(
            features=dino_on_out,
            coordinate_map_key=out_up.coordinate_map_key,
            tensor_stride=out_up.tensor_stride,
            coordinate_manager=out_up.coordinate_manager,
        )
        fused_in = me.cat(out_up, out_skip, dino_on_out)
        fused_out = fuse(fused_in)
        if getattr(self, '_dino_residual', False):
            return base_cat + fused_out
        return fused_out

    def forward(self, x, dino_feats=None, memory=None):
        dino_s1 = dino_s2 = dino_s4 = dino_s8 = dino_s16 = None
        if getattr(self, 'use_dino', False) and dino_feats is not None and len(dino_feats) >= 4:
            dino_s1 = dino_feats[0]
            dino_s2 = dino_feats[1]
            dino_s4 = dino_feats[2]
            dino_s8 = dino_feats[3]
            if len(dino_feats) >= 5:
                dino_s16 = dino_feats[4]

        out = self.conv0p1s1(x) # resulotion 1 * 0.02
        out = self.bn0(out)
        out_p1 = self.relu(out) 

        out = self.conv1p1s2(out_p1) # resulotion 2 * 0.02
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out) 

        out = self.conv2p2s2(out_b1p2) # resulotion 4 * 0.02
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out) 

        out = self.conv3p4s2(out_b2p4) # resulotion 8 * 0.02
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # pixel_dist=16
        out = self.conv4p8s2(out_b3p8) # resulotion 16 * 0.02
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        if memory is not None: # 对于不同尺度的特征基于memory进行融合
            out_b1p2_temp, out_b2p4_temp, out_b3p8_temp, out_temp = out_b1p2, out_b2p4, out_b3p8, out
            out_b1p2, out_b2p4, out_b3p8, out = memory([out_b1p2, out_b2p4, out_b3p8, out]) # 对于不同尺度的特征基于memory进行融合
            out_b1p2 = ME.SparseTensor(coordinate_map_key=out_b1p2_temp.coordinate_map_key, features=out_b1p2.features_at_coordinates(out_b1p2_temp.coordinates.float()), tensor_stride=out_b1p2_temp.tensor_stride, coordinate_manager=out_b1p2_temp.coordinate_manager)
            out_b2p4 = ME.SparseTensor(coordinate_map_key=out_b2p4_temp.coordinate_map_key, features=out_b2p4.features_at_coordinates(out_b2p4_temp.coordinates.float()), tensor_stride=out_b2p4_temp.tensor_stride, coordinate_manager=out_b2p4_temp.coordinate_manager)
            out_b3p8 = ME.SparseTensor(coordinate_map_key=out_b3p8_temp.coordinate_map_key, features=out_b3p8.features_at_coordinates(out_b3p8_temp.coordinates.float()), tensor_stride=out_b3p8_temp.tensor_stride, coordinate_manager=out_b3p8_temp.coordinate_manager)
            out = ME.SparseTensor(coordinate_map_key=out_temp.coordinate_map_key, features=out.features_at_coordinates(out_temp.coordinates.float()), tensor_stride=out_temp.tensor_stride, coordinate_manager=out_temp.coordinate_manager)

        # pixel_dist=8 进行上采样
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)
        if getattr(self, 'use_dino', False):
            out = self._fuse_with_dino(out, out_b3p8, dino_s8, self.dino_fuse_8)
        else:
            out = me.cat(out, out_b3p8)
        out = self.block5(out)

        # pixel_dist=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        if getattr(self, 'use_dino', False):
            out = self._fuse_with_dino(out, out_b2p4, dino_s4, self.dino_fuse_4)
        else:
            out = me.cat(out, out_b2p4)
        out = self.block6(out)

        # pixel_dist=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        if getattr(self, 'use_dino', False):
            out = self._fuse_with_dino(out, out_b1p2, dino_s2, self.dino_fuse_2)
        else:
            out = me.cat(out, out_b1p2)
        out = self.block7(out)

        # pixel_dist=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        if getattr(self, 'use_dino', False):
            out = self._fuse_with_dino(out, out_p1, dino_s1, self.dino_fuse_1)
        else:
            out = me.cat(out, out_p1)
        out = self.block8(out)

        return out

class Res16UNetBase_FF(Res16UNetBase):
    """Base class for Minkowski U-Net."""

    def forward(self, x, f=None, memory=None):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        if f is not None:
            out_p1 = f(out_p1)
        
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # pixel_dist=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        if memory is not None:
            out_b1p2_temp, out_b2p4_temp, out_b3p8_temp, out_temp = out_b1p2, out_b2p4, out_b3p8, out
            out_b1p2, out_b2p4, out_b3p8, out = memory([out_b1p2, out_b2p4, out_b3p8, out])
            out_b1p2 = ME.SparseTensor(coordinate_map_key=out_b1p2_temp.coordinate_map_key, features=out_b1p2.features_at_coordinates(out_b1p2_temp.coordinates.float()), tensor_stride=out_b1p2_temp.tensor_stride, coordinate_manager=out_b1p2_temp.coordinate_manager)
            out_b2p4 = ME.SparseTensor(coordinate_map_key=out_b2p4_temp.coordinate_map_key, features=out_b2p4.features_at_coordinates(out_b2p4_temp.coordinates.float()), tensor_stride=out_b2p4_temp.tensor_stride, coordinate_manager=out_b2p4_temp.coordinate_manager)
            out_b3p8 = ME.SparseTensor(coordinate_map_key=out_b3p8_temp.coordinate_map_key, features=out_b3p8.features_at_coordinates(out_b3p8_temp.coordinates.float()), tensor_stride=out_b3p8_temp.tensor_stride, coordinate_manager=out_b3p8_temp.coordinate_manager)
            out = ME.SparseTensor(coordinate_map_key=out_temp.coordinate_map_key, features=out.features_at_coordinates(out_temp.coordinates.float()), tensor_stride=out_temp.tensor_stride, coordinate_manager=out_temp.coordinate_manager)

        # pixel_dist=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)
        out = me.cat(out, out_b3p8)
        out = self.block5(out)

        # pixel_dist=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        out = me.cat(out, out_b2p4)
        out = self.block6(out)

        # pixel_dist=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        out = me.cat(out, out_b1p2)
        out = self.block7(out)

        # pixel_dist=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        out = me.cat(out, out_p1)
        out = self.block8(out)

        return out


class Res16UNet34(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

class Res16UNet34_FF(Res16UNetBase_FF):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

@MODELS.register_module()
class Res16UNet34C(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

@MODELS.register_module()
class Res16UNet34C_FF(Res16UNet34_FF):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)    
    
