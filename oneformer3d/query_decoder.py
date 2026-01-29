import torch
import torch.nn as nn
import pdb, time
import math
from typing import Optional
from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmdet3d.registry import MODELS
from torch_scatter import scatter_mean, scatter_add
from .mixformer3d import MultiScaleQuery
from torch.nn import functional as F

class CrossAttentionLayer(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources)): # batch_size
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z, _ = self.attn(y, y, y)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out


def _shift_scale_points(xyz: torch.Tensor, src_range):
    """Normalize xyz into [0,1] given (min,max) range.

    Args:
        xyz: (B, N, 3)
        src_range: (min_xyz, max_xyz) each (B, 1, 3) or (B, 3)
    """
    if src_range is None:
        return xyz
    src_min, src_max = src_range
    if src_min.ndim == 2:
        src_min = src_min.unsqueeze(1)
    if src_max.ndim == 2:
        src_max = src_max.unsqueeze(1)
    scale = (src_max - src_min).clamp(min=1e-6)
    return (xyz - src_min) / scale


class PositionEmbeddingCoordsSine(nn.Module):
    """Sine positional embedding for 3D coordinates.

    This is a minimal, dependency-free port of SegDINO3D's implementation.
    """

    def __init__(self, temperature=10000, normalize=True, scale=None, d_pos=256):
        super().__init__()
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.scale = float(scale) if scale is not None else 2 * math.pi
        self.d_pos = int(d_pos)

    def forward(self, xyz: torch.Tensor, input_range=None, modulated: Optional[torch.Tensor] = None):
        # xyz: (B, N, 3)
        assert xyz.ndim == 3 and xyz.shape[-1] == 3
        if self.normalize:
            xyz = _shift_scale_points(xyz, input_range)
        num_channels = self.d_pos

        ndim = num_channels // 3
        if ndim % 2 != 0:
            ndim -= 1
        rems = num_channels - (ndim * 3)
        assert ndim % 2 == 0

        final_embeds = []
        prev_dim = 0
        for d in range(3):
            cdim = ndim
            if rems > 0:
                cdim += 2
                rems -= 2
            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)
            raw_pos = xyz[:, :, d]
            raw_pos = raw_pos * self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        if modulated is not None:
            assert isinstance(modulated, torch.Tensor) and modulated.shape == xyz.shape
            for j in range(3):
                final_embeds[j] = final_embeds[j] * modulated[:, :, j:j + 1]

        return torch.cat(final_embeds, dim=2)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = int(num_layers)
        h = [hidden_dim] * (self.num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _safe_quantile_1d(x: torch.Tensor, q: float) -> float:
    """Quantile helper for small diagnostic tensors.

    Uses sort+index to avoid torch.quantile version differences across envs.
    """
    if x.numel() == 0:
        return 0.0
    x = x.detach().float().flatten()
    x_sorted, _ = torch.sort(x)
    idx = int(round((x_sorted.numel() - 1) * float(q)))
    idx = max(0, min(idx, x_sorted.numel() - 1))
    return float(x_sorted[idx].item())


def _daca_apply_debug_should_log(cfg: dict, seen: int) -> bool:
    dbg = cfg.get("apply_debug", {}) if isinstance(cfg, dict) else {}
    if not isinstance(dbg, dict) or not dbg.get("enable", False):
        return False
    log_first = int(dbg.get("log_first", 5))
    log_every = int(dbg.get("log_every", 200))
    if seen <= log_first:
        return True
    return (seen % max(log_every, 1)) == 0


def _daca_apply_debug_extra_stats(cfg: dict) -> bool:
    dbg = cfg.get("apply_debug", {}) if isinstance(cfg, dict) else {}
    if not isinstance(dbg, dict):
        return False
    return bool(dbg.get("extra_stats", True))


def _mean_over_layers(layer_stats, keys):
    if not layer_stats:
        return {}
    out = {}
    for k in keys:
        vals = []
        for d in layer_stats:
            v = d.get(k, None)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            out[k] = sum(vals) / float(len(vals))
    return out


class QueryDecoder(BaseModule):
    """Query decoder for SPFormer.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 objectness_flag, use_track_loss=False, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.fix_attention = fix_attention
        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        if num_instance_queries + num_semantic_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes + 1))
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.use_track_loss = use_track_loss
        if self.use_track_loss:
            self.out_track_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
    
    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'): # 是否有静态query
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i])) # [N_segments, 96] -> [N_segments, 256]
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)


@MODELS.register_module()
class ScanNetQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, **kwargs)
        assert num_semantic_linears in [1, 2]
        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)

    def _forward_head(self, queries, mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks = [], [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats, last_flag=True)
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats, last_flag)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            dict(
                cls_preds=cls_pred, sem_preds=sem_pred, masks=masks, scores=scores)
            for cls_pred, sem_pred, scores, masks in zip(
                cls_preds[:-1], sem_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)


@MODELS.register_module()
class ScanNetMixQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, in_channels, share_attn_mlp, share_mask_mlp,
                 cross_attn_mode, mask_pred_mode, temporal_attn=False, bbox_flag=False, 
                 use_query_memory2=False, query_stage=[0],use_track_loss=False, use_temporal_loss=False,
                 use_decouple=False, use_mot=False, mot_type=None, gdino_daca2d=None,
                 box3d_ca3d=None, track_window_stm=None, **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, in_channels=in_channels,use_track_loss=use_track_loss, **kwargs)
        assert num_semantic_linears in [1, 2]
        assert isinstance(cross_attn_mode, list)
        assert isinstance(mask_pred_mode, list) 
        assert mask_pred_mode[-1] == "P"

        self.cross_attn_mode = cross_attn_mode
        self.mask_pred_mode = mask_pred_mode
        self.temporal_attn = temporal_attn

        self.share_attn_mlp = share_attn_mlp
        if not share_attn_mlp:
            if "P" in self.cross_attn_mode:
                self.input_pts_proj = nn.Sequential(
                    nn.Linear(3 + in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())

        self.share_mask_mlp = share_mask_mlp
        if not share_mask_mlp:
            if "P" in self.mask_pred_mode:
                self.x_pts_mask = nn.Sequential(
                    nn.Linear(3 + in_channels, d_model), nn.ReLU(),
                    nn.Linear(d_model, d_model))

        self.bbox_flag = bbox_flag
        if self.bbox_flag:
            self.out_reg = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, 6))

        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)

        # ADDED
        self.use_query_memory2 = use_query_memory2
        if self.use_query_memory2:
            self.muti_scale_query = nn.ModuleList()
            self.muti_scale_norm = nn.ModuleList()
            self.query_stage = query_stage
            self.query_memory2 = [None for _ in range(len(self.query_stage))]
            self.pred_bbox_memory = [None for _ in range(len(self.query_stage))]
            for _ in range(len(self.query_stage)):
                self.muti_scale_query.append(MultiScaleQuery(embed_dims=256))
                self.muti_scale_norm.append(nn.LayerNorm(256))
        self.use_temporal_loss = use_temporal_loss
        if self.use_temporal_loss:
            self.merge_layer = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 256))
            # self.merge_layer = MergeLayer()
        self.use_mot = use_mot
        self.mot_type = mot_type
        if self.use_mot and mot_type == 'motr':
            self.motr_merge_layer = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 256))
        self.use_decouple = use_decouple
        if self.use_decouple:
            # self.query_inter = QueryInteractionX(256, 256)
            self.query_inter = MultiScaleQuery(embed_dims=256)
        # GDINO DACA-2D config (optional, default disabled)
        self.gdino_daca2d_cfg = gdino_daca2d or {}
        # Dedicated 2D query cross-attn layers (same depth as 3D decoder)
        self.dino_query_cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(self.d_model, self.num_heads, self.dropout, fix=self.fix_attention)
            for _ in range(len(self.cross_attn_layers))
        ])
        # Diagnostics: record whether DACA-2D actually changes 3D queries.
        self._daca_apply_seen = 0
        self._last_daca2d_apply_stats = None

        # SegDINO3D-style box-modulated CA-3D (optional, default disabled).
        # This is a lightweight approximation that adds (modulated) 3D positional
        # embeddings to SP-domain cross-attn inputs (layer0 only by default).
        self.box3d_ca_cfg = box3d_ca3d or {}
        self._box3d_ca_enabled = bool(self.box3d_ca_cfg.get('enable', False))
        if self._box3d_ca_enabled:
            temperature = float(self.box3d_ca_cfg.get('temperature', 10000.0))
            self.box3d_pe = PositionEmbeddingCoordsSine(
                temperature=temperature,
                normalize=True,
                d_pos=self.d_model,
            )
            self.box3d_ref_point_head = MLP(self.d_model, self.d_model, self.d_model, 2)
            self.box3d_ref_anchor_head = MLP(self.d_model, self.d_model, 3, 2)
            self.box3d_qpos_proj = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(len(self.cross_attn_layers))])
            self.box3d_kpos_proj = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(len(self.cross_attn_layers))])

        # Track-window STM (decoder-level; disabled by default).
        # This injects a distance-aware cross-attn from recent track prototypes
        # (LTM track bank) to current-frame instance queries.
        self.track_window_stm_cfg = track_window_stm or {}
        self._trk_stm_enabled = bool(self.track_window_stm_cfg.get("enable", False))
        # Diagnostic collection flag (set externally, e.g. by model.predict when
        # online_monitor is enabled). Keep False by default to avoid overhead.
        self._trk_stm_diag_collect = False
        self._last_trk_stm_apply_stats = None
        if self._trk_stm_enabled:
            self._trk_stm_window = max(int(self.track_window_stm_cfg.get("window", 5)), 1)
            self._trk_stm_mode = str(self.track_window_stm_cfg.get("mode", "scale")).lower()
            if self._trk_stm_mode not in ("cross", "distance", "scale"):
                self._trk_stm_mode = "scale"
            # Use `dist_lambda` (preferred) or legacy `lambda` (avoid reserved keyword in configs).
            self._trk_stm_lambda = float(
                self.track_window_stm_cfg.get("dist_lambda", self.track_window_stm_cfg.get("lambda", 1.0))
            )
            # Identity-safe residual gating (recommended for finetune). The STM
            # branch is randomly initialized when loading old checkpoints, so
            # gating lets the model start close to baseline and gradually learn
            # to use track memory.
            gate_init = float(self.track_window_stm_cfg.get("gate_init", -6.0))  # sigmoid(-6)≈0.0025
            try:
                n_layers = int(getattr(self, "num_layers", len(self.cross_attn_layers)))
            except Exception:
                n_layers = len(self.cross_attn_layers)
            n_layers = max(int(n_layers), 1)
            self._trk_stm_gate = nn.Parameter(torch.full((n_layers,), gate_init, dtype=torch.float32))

            self._trk_stm_norm = nn.LayerNorm(self.d_model)
            self._trk_stm_dropout = nn.Dropout(self.dropout)
            self._trk_stm_mha = nn.MultiheadAttention(
                self.d_model, self.num_heads, dropout=self.dropout, batch_first=True
            )
            self._trk_stm_msq = MultiScaleQuery(
                embed_dims=self.d_model, num_heads=self.num_heads, dropout=self.dropout
            )
            try:
                self._trk_stm_msq.init_weights()
            except Exception:
                pass

    def _apply_track_window_stm(self, queries, track_instances, query3d_pos, layer_idx: int = -1):
        """Apply track-window STM to current-frame queries (per batch element).

        Args:
            queries: List[Tensor], each (Nq, D)
            track_instances: Instances with fields queries/bboxes/valid_track/disappear_time
            query3d_pos: List[Tensor] or Tensor, each (Nq, 3)
        """
        if not bool(getattr(self, "_trk_stm_enabled", False)):
            return queries
        if track_instances is None or query3d_pos is None:
            return queries

        if torch.is_tensor(query3d_pos):
            qpos_list = [query3d_pos]
        elif isinstance(query3d_pos, (list, tuple)):
            qpos_list = list(query3d_pos)
        else:
            return queries

        if len(qpos_list) < len(queries):
            return queries

        diag_collect = bool(getattr(self, "_trk_stm_diag_collect", False))
        layer_stats = None
        n_valid_batches = 0
        if diag_collect:
            layer_stats = {
                "layer": int(layer_idx),
                "mode": str(getattr(self, "_trk_stm_mode", "scale")),
                "window": int(getattr(self, "_trk_stm_window", 1)),
                "dist_lambda": float(getattr(self, "_trk_stm_lambda", 1.0)),
                "gate_alpha": 0.0,
                "mem_tracks_total_mean": 0.0,
                "mem_tracks_win_mean": 0.0,
                "applied_q_mean": 0.0,
                "delta_rel_mean": 0.0,
                "delta_rel_p50": 0.0,
                "delta_rel_p90": 0.0,
                "min_dist_p50": 0.0,
                "min_dist_p90": 0.0,
                "delta_nan": 0.0,
            }

        out = []
        # Gate (per decoder layer). Use sigmoid so the default init is
        # near-zero (identity), but remains learnable.
        try:
            gate = getattr(self, "_trk_stm_gate", None)
            if torch.is_tensor(gate) and gate.numel() > 0:
                idx = int(layer_idx) if int(layer_idx) >= 0 else (int(gate.numel()) - 1)
                idx = max(0, min(idx, int(gate.numel()) - 1))
                alpha = gate[idx].sigmoid().to(dtype=queries[0].dtype if queries and torch.is_tensor(queries[0]) else torch.float32)
            else:
                alpha = None
        except Exception:
            alpha = None

        for b in range(len(queries)):
            q = queries[b]
            qpos = qpos_list[b]
            if not (torch.is_tensor(q) and torch.is_tensor(qpos)):
                out.append(q)
                continue
            if q.numel() == 0 or qpos.numel() == 0:
                out.append(q)
                continue
            if qpos.shape[0] != q.shape[0] or qpos.shape[-1] < 3:
                out.append(q)
                continue

            try:
                trk_q_all = track_instances.queries[b]
                trk_box_all = track_instances.bboxes[b]
            except Exception:
                out.append(q)
                continue

            if not (torch.is_tensor(trk_q_all) and torch.is_tensor(trk_box_all)):
                out.append(q)
                continue
            if trk_q_all.numel() == 0 or trk_box_all.numel() == 0:
                out.append(q)
                continue
            if trk_box_all.shape[0] != trk_q_all.shape[0]:
                out.append(q)
                continue

            try:
                valid = track_instances.valid_track[b]
                if not torch.is_tensor(valid):
                    valid = None
            except Exception:
                valid = None
            if valid is None:
                valid = torch.ones((trk_q_all.shape[0],), dtype=torch.bool, device=trk_q_all.device)
            else:
                valid = valid.to(device=trk_q_all.device)

            try:
                disp = track_instances.disappear_time[b]
                if not torch.is_tensor(disp):
                    disp = None
            except Exception:
                disp = None
            if disp is None:
                mem_mask = valid
            else:
                disp = disp.to(device=trk_q_all.device)
                mem_mask = valid & (disp <= int(self._trk_stm_window - 1))

            # Optional: memory token dropout (train-time only). This encourages
            # the model to not overly rely on a few near-by tracks and can help
            # the STM gate learn to open when memory is informative.
            try:
                drop_p = float(self.track_window_stm_cfg.get("mem_dropout", 0.0))
            except Exception:
                drop_p = 0.0
            if bool(getattr(self, "training", False)) and drop_p > 0.0:
                drop_p = max(0.0, min(drop_p, 0.95))
                try:
                    idx = torch.nonzero(mem_mask, as_tuple=True)[0]
                    if idx.numel() > 1:
                        keep = torch.rand((idx.numel(),), device=idx.device) > drop_p
                        if keep.any():
                            mem_mask = torch.zeros_like(mem_mask)
                            mem_mask[idx[keep]] = True
                        else:
                            # Keep at least one token to avoid degenerate no-op.
                            j = int(torch.randint(0, idx.numel(), (1,), device=idx.device).item())
                            mem_mask = torch.zeros_like(mem_mask)
                            mem_mask[idx[j]] = True
                except Exception:
                    pass

            if int(mem_mask.sum().item()) <= 0:
                out.append(q)
                continue

            mem_q = trk_q_all[mem_mask].detach()
            mem_pos = trk_box_all[mem_mask][:, :3].detach()
            if mem_q.numel() == 0 or mem_pos.numel() == 0:
                out.append(q)
                continue

            mode = str(getattr(self, "_trk_stm_mode", "scale")).lower()
            delta = None
            delta_nan = False
            if mode == "cross":
                attn_out, _ = self._trk_stm_mha(
                    q.unsqueeze(0),
                    mem_q.unsqueeze(0),
                    mem_q.unsqueeze(0),
                    attn_mask=None,
                )
                delta = attn_out.squeeze(0)
            elif mode == "distance":
                try:
                    dist = torch.cdist(qpos[:, :3], mem_pos[:, :3], p=2)
                    attn_bias = (-dist * float(getattr(self, "_trk_stm_lambda", 1.0))).to(
                        dtype=q.dtype, device=q.device
                    )
                except Exception:
                    out.append(q)
                    continue
                attn_out, _ = self._trk_stm_mha(
                    q.unsqueeze(0),
                    mem_q.unsqueeze(0),
                    mem_q.unsqueeze(0),
                    attn_mask=attn_bias,
                )
                delta = attn_out.squeeze(0)
            else:  # "scale"
                lam = float(getattr(self, "_trk_stm_lambda", 1.0))
                try:
                    delta = self._trk_stm_msq(
                        [qpos[:, :3].to(device=q.device, dtype=q.dtype)],
                        [q],
                        [mem_q.to(device=q.device, dtype=q.dtype)],
                        [mem_q.to(device=q.device, dtype=q.dtype)],
                        [mem_pos[:, :3].to(device=q.device, dtype=q.dtype)],
                        dist_scale=lam,
                    )[0]
                except Exception:
                    out.append(q)
                    continue

            if delta is None or (not torch.is_tensor(delta)) or delta.shape != q.shape:
                out.append(q)
                continue
            try:
                delta_nan = bool(torch.isnan(delta).any().item()) or bool(torch.isinf(delta).any().item())
            except Exception:
                delta_nan = False

            # Apply residual gate: q <- q + alpha * delta (alpha≈0 at init).
            if alpha is None:
                applied_delta = delta
                alpha_f = 1.0
            else:
                applied_delta = delta * alpha.to(device=q.device, dtype=q.dtype)
                alpha_f = float(alpha.detach().cpu().item())

            if diag_collect and layer_stats is not None:
                try:
                    valid_sum = int(valid.sum().item()) if torch.is_tensor(valid) else int(trk_q_all.shape[0])
                    win_sum = int(mem_mask.sum().item())
                    layer_stats["mem_tracks_total_mean"] += float(valid_sum)
                    layer_stats["mem_tracks_win_mean"] += float(win_sum)
                    layer_stats["applied_q_mean"] += float(q.shape[0])
                    layer_stats["gate_alpha"] += float(alpha_f)

                    denom = q.norm(dim=-1).clamp(min=1e-6)
                    rel = applied_delta.norm(dim=-1) / denom
                    rel = rel.masked_fill(~torch.isfinite(rel), 0.0)
                    layer_stats["delta_rel_mean"] += float(rel.mean().item())
                    layer_stats["delta_rel_p50"] += _safe_quantile_1d(rel, 0.5)
                    layer_stats["delta_rel_p90"] += _safe_quantile_1d(rel, 0.9)

                    d = torch.cdist(qpos[:, :3], mem_pos[:, :3], p=2)
                    dmin = d.min(dim=1).values
                    dmin = dmin.masked_fill(~torch.isfinite(dmin), float("inf"))
                    layer_stats["min_dist_p50"] += _safe_quantile_1d(dmin, 0.5)
                    layer_stats["min_dist_p90"] += _safe_quantile_1d(dmin, 0.9)

                    layer_stats["delta_nan"] += 1.0 if delta_nan else 0.0
                    n_valid_batches += 1
                except Exception:
                    pass

            q_new = self._trk_stm_norm(q + self._trk_stm_dropout(applied_delta))
            out.append(q_new)

        if diag_collect and layer_stats is not None and n_valid_batches > 0:
            for k in (
                "gate_alpha",
                "mem_tracks_total_mean",
                "mem_tracks_win_mean",
                "applied_q_mean",
                "delta_rel_mean",
                "delta_rel_p50",
                "delta_rel_p90",
                "min_dist_p50",
                "min_dist_p90",
                "delta_nan",
            ):
                layer_stats[k] = float(layer_stats[k]) / float(n_valid_batches)
            try:
                if not hasattr(self, "_trk_stm_apply_layer_stats") or self._trk_stm_apply_layer_stats is None:
                    self._trk_stm_apply_layer_stats = []
                self._trk_stm_apply_layer_stats.append(layer_stats)
            except Exception:
                pass

        return out
    def reset_decouple(self):
        """Reset the decouple module.
        """
        if self.use_decouple:
            self.before_query_memory = None
            self.before_query_boxes = None

    def _daca_layer_enabled(self, layer_idx, attn_mask, cfg):
        """Check if DACA-2D should run at this layer (SP-domain only)."""
        if not isinstance(cfg, dict):
            return False
        if cfg.get("inject_domain", "sp") != "sp":
            return False
        inject_layers = cfg.get("inject_layers", "auto_sp")
        if inject_layers == "auto_sp":
            return self.mask_pred_mode[layer_idx] == "SP"
        if isinstance(inject_layers, (list, tuple)):
            return layer_idx in inject_layers
        return False

    def _box3d_ca_enabled_for_layer(self, layer_idx: int) -> bool:
        if not self._box3d_ca_enabled:
            return False
        cfg = self.box3d_ca_cfg if isinstance(self.box3d_ca_cfg, dict) else {}
        layers = cfg.get("layers", [0])
        if layers == "all":
            return True
        if isinstance(layers, (list, tuple, set)):
            return int(layer_idx) in set(int(x) for x in layers)
        return int(layer_idx) == 0

    def _apply_box3d_ca3d_sp(self, layer_idx: int, inst_feats, queries, attn_mask,
                              sp_pos_list_elastic, scene_range,
                              base_query_pos, ref_sizes_norm):
        """SegDINO3D-style box-modulated CA on SP-domain (approximation).

        This modulates SP cross-attn by adding (optionally size-modulated) 3D
        positional embeddings to queries/keys before attention.

        Args are lists (len B). Returns updated queries and updated ref_sizes_norm.
        """
        if not self._box3d_ca_enabled_for_layer(layer_idx):
            return queries, ref_sizes_norm
        if sp_pos_list_elastic is None or scene_range is None or base_query_pos is None:
            return queries, ref_sizes_norm
        if not (isinstance(sp_pos_list_elastic, (list, tuple)) and isinstance(scene_range, (list, tuple)) and isinstance(base_query_pos, (list, tuple))):
            return queries, ref_sizes_norm
        if len(sp_pos_list_elastic) < len(queries) or len(scene_range) < len(queries) or len(base_query_pos) < len(queries):
            return queries, ref_sizes_norm

        cfg = self.box3d_ca_cfg if isinstance(self.box3d_ca_cfg, dict) else {}
        use_mod = bool(cfg.get("use_modulation", True))
        eps = float(cfg.get("eps", 1e-6))
        max_mod = float(cfg.get("max_mod", 50.0))

        out = []
        for b in range(len(queries)):
            q = queries[b]
            k = inst_feats[b]
            if q.numel() == 0 or k.numel() == 0:
                out.append(q)
                continue
            sp_pos = sp_pos_list_elastic[b]
            qpos_base = base_query_pos[b]
            if not (torch.is_tensor(sp_pos) and torch.is_tensor(qpos_base)):
                out.append(q)
                continue
            if sp_pos.shape[0] != k.shape[0]:
                out.append(q)
                continue
            if qpos_base.shape[0] != q.shape[0]:
                out.append(q)
                continue
            smin, smax = scene_range[b]
            if not (torch.is_tensor(smin) and torch.is_tensor(smax)):
                out.append(q)
                continue
            smin = smin.to(device=q.device, dtype=q.dtype)
            smax = smax.to(device=q.device, dtype=q.dtype)
            span = (smax - smin).clamp(min=1e-6)

            # Query sizes in normalized space (for modulation); default to 0.5 if absent.
            if ref_sizes_norm is None or ref_sizes_norm[b] is None:
                qsize_n = torch.full_like(qpos_base, 0.5)
            else:
                qsize_n = ref_sizes_norm[b].to(device=q.device, dtype=q.dtype)
                if qsize_n.shape != qpos_base.shape:
                    qsize_n = torch.full_like(qpos_base, 0.5)

            modulated = None
            if use_mod:
                ref = torch.sigmoid(self.box3d_ref_anchor_head(q))  # (nq,3)
                modulated = (ref / (qsize_n + eps)).clamp(max=max_mod)

            q_pe = self.box3d_pe(qpos_base.unsqueeze(0), input_range=(smin.unsqueeze(0), smax.unsqueeze(0)),
                                 modulated=modulated.unsqueeze(0) if modulated is not None else None)[0]
            q_pe = self.box3d_ref_point_head(q_pe)
            k_pe = self.box3d_pe(sp_pos.unsqueeze(0), input_range=(smin.unsqueeze(0), smax.unsqueeze(0)))[0]

            q_in = q + self.box3d_qpos_proj[layer_idx](q_pe)
            k_in = k + self.box3d_kpos_proj[layer_idx](k_pe)

            attn = self.cross_attn_layers[layer_idx].attn
            attn_b = attn_mask[b] if attn_mask is not None else None
            attn_out, _ = attn(q_in, k_in, k, attn_mask=attn_b)
            if self.cross_attn_layers[layer_idx].fix:
                attn_out = self.cross_attn_layers[layer_idx].dropout(attn_out)
            attn_out = attn_out + q
            if self.cross_attn_layers[layer_idx].fix:
                attn_out = self.cross_attn_layers[layer_idx].norm(attn_out)
            out.append(attn_out)
        return out, ref_sizes_norm

    def _apply_daca2d(self, queries, attn_mask, query2d_feats, query2d_pos,
                       sp_pos_list, cfg, layer_idx: int):
        """Apply DACA-2D cross-attention (SP-domain only).

        Args:
            queries: List[Tensor] length B, each (Nq3d, D).
            attn_mask: List[Tensor] length B, each (Nq3d, Nsp) bool.
            query2d_feats: List[Tensor] length B, each (Nq2d, D).
            query2d_pos: List[Tensor] length B, each (Nq2d, 3).
            sp_pos_list: List[Tensor] length B, each (Nsp, 3).
            cfg: gdino_daca2d config dict.
        """
        if not self._daca_layer_enabled(layer_idx, attn_mask, cfg):
            return queries
        if attn_mask is None or sp_pos_list is None:
            return queries
        if not (isinstance(query2d_feats, (list, tuple)) and isinstance(query2d_pos, (list, tuple))):
            return queries
        if len(query2d_feats) < len(queries) or len(query2d_pos) < len(queries):
            return queries

        mask_cfg = cfg.get("mask", {}) if isinstance(cfg, dict) else {}
        thr = float(mask_cfg.get("thr", 0.2))
        metric = str(mask_cfg.get("metric", "l1")).lower()
        p = 1 if metric == "l1" else 2
        extra_stats = _daca_apply_debug_extra_stats(cfg)

        outputs = []
        layer_stats = {
            "layer": int(layer_idx),
            "metric": metric,
            "thr": float(thr),
            "delta_rel_mean": 0.0,
            "allowed_q2d_mean": 0.0,
            "allowed_q2d_zero_rate": 0.0,
            "allowed_q2d_p50": 0.0,
            "allowed_q2d_p90": 0.0,
            "nq3d_mean": 0.0,
            "nq2d_mean": 0.0,
            "sp_allow_mean": 0.0,
            "sp_allow_p50": 0.0,
            "sp_allow_p90": 0.0,
            "min_dist_q3d_p50": 0.0,
            "min_dist_q3d_p90": 0.0,
            "min_dist_q2d_p50": 0.0,
            "min_dist_q2d_p90": 0.0,
            "q2d_any_sp_rate": 0.0,
        }
        n_valid_batches = 0
        for b in range(len(queries)):
            attn_b = attn_mask[b]
            sp_pos = sp_pos_list[b]
            q2d = query2d_feats[b]
            q2d_pos = query2d_pos[b]
            if not torch.is_tensor(attn_b) or not torch.is_tensor(sp_pos):
                outputs.append(queries[b])
                continue
            if not torch.is_tensor(q2d) or not torch.is_tensor(q2d_pos):
                outputs.append(queries[b])
                continue
            if attn_b.numel() == 0 or sp_pos.numel() == 0 or q2d.numel() == 0 or q2d_pos.numel() == 0:
                outputs.append(queries[b])
                continue
            # Ensure SP mask dimension matches sp_pos
            if attn_b.shape[1] != sp_pos.shape[0]:
                outputs.append(queries[b])
                continue

            dist = torch.cdist(sp_pos, q2d_pos, p=p)
            reach = (~attn_b).float() @ (dist < thr).float()
            daca_mask = (reach == 0)
            # Diagnostic: how many 2D queries are reachable per 3D query (before dummy append).
            try:
                allowed = (~daca_mask).sum(dim=1).float()  # (Nq3d,)
                layer_stats["allowed_q2d_mean"] += float(allowed.mean().item())
                layer_stats["allowed_q2d_zero_rate"] += float((allowed == 0).float().mean().item())
                layer_stats["allowed_q2d_p50"] += _safe_quantile_1d(allowed, 0.5)
                layer_stats["allowed_q2d_p90"] += _safe_quantile_1d(allowed, 0.9)
                layer_stats["nq3d_mean"] += float(attn_b.shape[0])
                layer_stats["nq2d_mean"] += float(q2d.shape[0])
                if extra_stats:
                    sp_allow = (~attn_b).sum(dim=1).float()  # (Nq3d,)
                    layer_stats["sp_allow_mean"] += float(sp_allow.mean().item())
                    layer_stats["sp_allow_p50"] += _safe_quantile_1d(sp_allow, 0.5)
                    layer_stats["sp_allow_p90"] += _safe_quantile_1d(sp_allow, 0.9)

                    # For each 3D query: min dist to any 2D query among its allowed SPs (approx).
                    # 1) per-SP min dist to any q2d
                    dist_sp_min = dist.min(dim=1).values  # (Nsp,)
                    # 2) per-q3d min over allowed SPs
                    allowed_sp = ~attn_b  # (Nq3d, Nsp)
                    masked = dist_sp_min.unsqueeze(0).expand_as(allowed_sp.float())
                    masked = masked.masked_fill(~allowed_sp, float("inf"))
                    min_dist_q3d = masked.min(dim=1).values
                    min_dist_q3d = min_dist_q3d.masked_fill(~torch.isfinite(min_dist_q3d), float("inf"))
                    layer_stats["min_dist_q3d_p50"] += _safe_quantile_1d(min_dist_q3d, 0.5)
                    layer_stats["min_dist_q3d_p90"] += _safe_quantile_1d(min_dist_q3d, 0.9)

                    # For each 2D query: min dist to any SP (global, independent of attn mask)
                    dist_q2d_min = dist.min(dim=0).values  # (Nq2d,)
                    layer_stats["min_dist_q2d_p50"] += _safe_quantile_1d(dist_q2d_min, 0.5)
                    layer_stats["min_dist_q2d_p90"] += _safe_quantile_1d(dist_q2d_min, 0.9)
                    layer_stats["q2d_any_sp_rate"] += float((dist_q2d_min < thr).float().mean().item())
                n_valid_batches += 1
            except Exception:
                pass
            # Append dummy query to avoid empty attention rows (SegDINO3D style)
            q2d = torch.cat([q2d, q2d.new_ones(1, q2d.shape[1])], dim=0)
            daca_mask = torch.cat([daca_mask, daca_mask.new_zeros(daca_mask.shape[0], 1)], dim=1)

            q3d_in = queries[b]
            out_b = self.dino_query_cross_attn_layers[layer_idx]([q2d], [q3d_in], [daca_mask])[0]
            # Diagnostic: relative change magnitude (||delta|| / ||q||).
            try:
                delta = out_b - q3d_in
                denom = q3d_in.norm(dim=-1).mean().clamp(min=1e-6)
                rel = delta.norm(dim=-1).mean() / denom
                layer_stats["delta_rel_mean"] += float(rel.item())
            except Exception:
                pass
            outputs.append(out_b)

        if n_valid_batches > 0:
            for k in (
                "delta_rel_mean",
                "allowed_q2d_mean",
                "allowed_q2d_zero_rate",
                "allowed_q2d_p50",
                "allowed_q2d_p90",
                "nq3d_mean",
                "nq2d_mean",
                "sp_allow_mean",
                "sp_allow_p50",
                "sp_allow_p90",
                "min_dist_q3d_p50",
                "min_dist_q3d_p90",
                "min_dist_q2d_p50",
                "min_dist_q2d_p90",
                "q2d_any_sp_rate",
            ):
                layer_stats[k] = float(layer_stats[k]) / float(n_valid_batches)
        # Stash per-layer stats for the outer forward to aggregate/log.
        try:
            if not hasattr(self, "_daca_apply_layer_stats") or self._daca_apply_layer_stats is None:
                self._daca_apply_layer_stats = []
            self._daca_apply_layer_stats.append(layer_stats)
        except Exception:
            pass
        return outputs
    
    def reset_query_memory2(self):
        """Reset the detector.
        """
        if self.use_query_memory2:
            self.query_memory2 = [None for _ in range(len(self.query_stage))]
            self.pred_bbox_memory = [None for _ in range(len(self.query_stage))]

    def _forward_head(self, queries, mask_feats, mask_pts_feats, last_flag, layer):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks, pred_bboxes = [], [], [], [], [], []
        object_queries = []
        for i in range(len(queries)): # batch_size
            norm_query = self.out_norm(queries[i]) # [N_segments, 256]
            object_queries.append(norm_query)
            cls_preds.append(self.out_cls(norm_query)) # 每个的类别 [N_segments, 256] —> [N_segments, 2(CA 200) or 19]
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag else None # None
            # pred_score = self.out_track_score(norm_query) if self.use_track_loss else pred_score
            pred_scores.append(pred_score) # 这个不存在
            if self.bbox_flag: # False (True in CA 200)
                reg_final = self.out_reg(norm_query) # [N_segments, 256] -> [N_segments, 6]
                reg_distance = torch.exp(reg_final[:, 3:6])
                pred_bbox = torch.cat([reg_final[:, :3], reg_distance], dim=1)
            else: pred_bbox = None
            pred_bboxes.append(pred_bbox)
            if self.mask_pred_mode[layer] == "SP": # ['SP', 'SP', 'P', 'P']
                pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i]) # ? 计算点积 [N_q, N_seg] 为啥数量不一样, * (0.5 ~ 1) train的时候有query选择的部分
            elif self.mask_pred_mode[layer] == "P":
                pred_mask = torch.einsum('nd,md->nm', norm_query, mask_pts_feats[i]) # ? [61, 20000] 这个mask是完全基于特征相似度的吗？ 
            else:
                raise NotImplementedError("Query decoder not implemented!")
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool() # [N_q, N_seg] -> [N_q, N_seg]
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False # 如果所有键位置都被遮蔽，查询将无法与任何键进行交互，导致注意力权重无法正常计算。这可能会引发数值问题，如 NaN（非数字）值。
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks, object_queries, pred_bboxes

    def forward_iter_pred(self, sp_feats, p_feats, queries, super_points, prev_queries=None, use_temporal_loss=False,
                          inst_dict=None, track_instances=None, use_one2many=False,
                          query2d_feats=None, query2d_pos=None, gdino_daca2d_cfg=None,
                          sp_pos_list_override=None,
                          sp_pos_list_elastic=None, scene_range=None, query3d_pos=None):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        # Note: sp_pos_list_elastic/scene_range/query3d_pos are reserved for
        # SegDINO3D-style box-modulated CA-3D. They are accepted here for
        # forward-compatibility and are no-ops unless the corresponding feature
        # is enabled in this decoder.
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        object_queries, pred_bboxes = [], []
        inst_feats = [self.input_proj(y) for y in sp_feats] if "SP" in self.cross_attn_mode else None # [N_segments, 96] -> [N_segments, 256]
        if use_temporal_loss and inst_dict is not None:
            before_features = inst_dict['querys']
            before_features_idx_list = inst_dict['best_obj_ids_list']
            # for i in range(len(inst_feats)):
            #     inst_feats[i][before_features_idx_list[i]] = self.merge_layer(torch.cat([inst_feats[i][before_features_idx_list[i]], before_features[i]], dim=1))

            for i in range(len(inst_feats)):
                # 取出原来的 inst_feats[i]
                old_feats = inst_feats[i]
                # 计算需要替换进去的 merged_part
                merged_part = self.merge_layer(
                    torch.cat([old_feats[before_features_idx_list[i]], before_features[i]], dim=1)
                )

                # 克隆出一个新的张量
                new_feats = old_feats.clone()
                # 在克隆的张量上进行索引赋值
                new_feats[before_features_idx_list[i]] = merged_part

                # 将更新后的张量放回 inst_feats[i]
                inst_feats[i] = new_feats
        if track_instances is not None:
            pass

        inst_pts_feats = [self.input_proj(y) if self.share_attn_mlp else self.input_pts_proj(y)
             for y in p_feats] if "P" in self.cross_attn_mode else None # None
        mask_feats = [self.x_mask(y) for y in sp_feats] if "SP" in self.mask_pred_mode else None # [N_segments, 96] -> [N_segments, 256]
        mask_pts_feats = [self.x_mask(y) if self.share_mask_mlp else self.x_pts_mask(y)
             for y in p_feats] if "P" in self.mask_pred_mode else None # [20000, 99] -> [20000, 256]
        queries = self._get_queries(queries, len(sp_feats)) # [N_segments, 96] -> [N_segments, 256] # ! 这个和inst_feats差不多？

        # Reset per-forward track-window STM stats when diagnostics are enabled.
        if bool(getattr(self, "_trk_stm_enabled", False)) and bool(getattr(self, "_trk_stm_diag_collect", False)):
            try:
                self._trk_stm_apply_layer_stats = []
            except Exception:
                pass

        # Box-modulated CA-3D bookkeeping (SegDINO3D-style, optional).
        # qpos_ref_list/qsize_ref_norm are updated per-layer from current bbox predictions.
        qpos_base_list = None
        qpos_ref_list = None
        qsize_ref_norm = None
        if self._box3d_ca_enabled and isinstance(scene_range, (list, tuple)) and query3d_pos is not None:
            if torch.is_tensor(query3d_pos):
                qpos_base_list = [query3d_pos]
            else:
                qpos_base_list = list(query3d_pos) if isinstance(query3d_pos, (list, tuple)) else None
            if qpos_base_list is not None:
                qpos_ref_list = list(qpos_base_list)
                qsize_ref_norm = []
                for b in range(len(queries)):
                    if b >= len(qpos_ref_list) or not torch.is_tensor(qpos_ref_list[b]):
                        qsize_ref_norm.append(None)
                        continue
                    qsize_ref_norm.append(torch.full_like(qpos_ref_list[b], 0.5))
        # Resolve GDINO DACA-2D config (optional, default disabled)
        gdino_cfg = gdino_daca2d_cfg or self.gdino_daca2d_cfg or {}
        use_daca2d = bool(gdino_cfg.get("enable", False)) and query2d_feats is not None and query2d_pos is not None
        if use_daca2d:
            # Reset per-forward stats container (avoid cross-iter accumulation).
            self._daca_apply_layer_stats = []
            # Normalize query2d inputs into per-batch lists.
            if torch.is_tensor(query2d_feats):
                query2d_feats = [query2d_feats]
            if torch.is_tensor(query2d_pos):
                query2d_pos = [query2d_pos]
            if not (isinstance(query2d_feats, (list, tuple)) and isinstance(query2d_pos, (list, tuple))):
                use_daca2d = False
        # Precompute SP positions when needed (only for SP-domain DACA)
        sp_pos_list = None
        if use_daca2d:
            try:
                if sp_pos_list_override is not None:
                    sp_pos_list = sp_pos_list_override
                else:
                    sp_pos_list = []
                    for pf, sp in zip(p_feats, super_points[0]):
                        xyz = pf[:, :3]
                        sp_id = sp.to(xyz.device)
                        sp_pos = scatter_mean(xyz, sp_id, dim=0)
                        sp_pos_list.append(sp_pos)
            except Exception:
                sp_pos_list = None
                use_daca2d = False
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask, object_query, pred_bbox = \
             self._forward_head(queries, mask_feats, mask_pts_feats, last_flag=False, layer=0)
        cls_preds.append(cls_pred) # [N_segments, 2]
        sem_preds.append(sem_pred) # None
        pred_scores.append(pred_score) # None
        pred_masks.append(pred_mask) # [N_segments, N_segments]
        object_queries.append(object_query) # [N_segments, 256]
        pred_bboxes.append(pred_bbox) # [N_segments, 6]

            
        if use_one2many:
            one2many_output_dict = {'cls_preds':[], 'sem_preds':[], 'scores':[], 'masks':[], 'queries':[], 'pred_bboxes':[]}
            one2many_output_dict['cls_preds'].append(cls_pred)
            one2many_output_dict['sem_preds'].append(sem_pred)
            one2many_output_dict['scores'].append(pred_score)
            one2many_output_dict['masks'].append(pred_mask)
            one2many_output_dict['queries'].append(object_query)
            one2many_output_dict['pred_bboxes'].append(pred_bbox)
            one2many_queries= []
            one2many_attn_mask = []
            for i in range(len(queries)):
                one2many_queries.append(queries[i].clone())
                one2many_attn_mask.append(attn_mask[i].clone())

        for i in range(len(self.cross_attn_layers)): # 3 [queries查询inst_feats] [queries查询inst_feats] [queries查询inst_feats]
            if self.cross_attn_mode[i+1] == "SP" and self.mask_pred_mode[i] == "SP": # SP 内的attention，使用mask
                if self._box3d_ca_enabled and qpos_ref_list is not None and qsize_ref_norm is not None:
                    queries, _ = self._apply_box3d_ca3d_sp(
                        i, inst_feats, queries, attn_mask,
                        sp_pos_list_elastic=sp_pos_list_elastic,
                        scene_range=scene_range,
                        base_query_pos=qpos_ref_list,
                        ref_sizes_norm=qsize_ref_norm,
                    )
                else:
                    queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask) # K Q mask [N_segments, 256], [N_segments, 256], [N_segments, N_segments] -> [N_segments, 256]
                if use_one2many:
                    one2many_queries = self.cross_attn_layers[i](inst_feats, one2many_queries, one2many_attn_mask)
            elif self.cross_attn_mode[i+1] == "SP" and self.mask_pred_mode[i] == "P":   # current method, change P mask to SP
                if not use_temporal_loss:
                    xyz_weights = torch.chunk(super_points[1], len(super_points[0]), dim=0) # torch.chunk(input, chunks, dim=0) 会把输入的张量 input 按照指定的维度 dim 和指定的分块数 chunks 来分割。
                    attn_mask_score = [scatter_mean(att.float() * xyz_w.view(1, -1), sp, dim=1) # [20000, 1]
                        for att, sp, xyz_w in zip(attn_mask, super_points[0], xyz_weights)] # 将点的mask转换为超点的mask
                    attn_mask = [(att > 0.5).bool() for att in attn_mask_score] # > 0.5, not <
                    # If attn_mask has all-True row, the result of CA will be nan 所有最小的attn对应的都设定为False
                    for j in range(len(attn_mask)): # batch_size
                        mask = ~(attn_mask_score[j] == attn_mask_score[j].min(dim=1, keepdim=True)[0])
                        attn_mask[j] *= mask
                else:
                    # attn_mask = None
                    xyz_weights = torch.chunk(super_points[1], len(super_points[0]), dim=0) # torch.chunk(input, chunks, dim=0) 会把输入的张量 input 按照指定的维度 dim 和指定的分块数 chunks 来分割。
                    attn_mask_score = [scatter_mean(att.float() * xyz_w.view(1, -1), sp, dim=1) # [20000, 1]
                        for att, sp, xyz_w in zip(attn_mask, super_points[0], xyz_weights)] # 将点的mask转换为超点的mask
                    attn_mask = [(att > 0.5).bool() for att in attn_mask_score] # > 0.5, not <
                    # If attn_mask has all-True row, the result of CA will be nan 所有最小的attn对应的都设定为False
                    for j in range(len(attn_mask)): # batch_size
                        mask = ~(attn_mask_score[j] == attn_mask_score[j].min(dim=1, keepdim=True)[0])
                        attn_mask[j] *= mask
                    
                    # if attn_mask[0].shape[1] != inst_feats[0].shape[0]:
                    #     attn_mask = [torch.cat([att, torch.zeros(att.shape[0], inst_feats[0].shape[0] - att.shape[1], device=att.device).bool()], dim=1) for att in attn_mask]
                    for j in range(len(attn_mask)):
                        if attn_mask[j].shape[1] != inst_feats[j].shape[0]:
                            attn_mask[j] = torch.cat([attn_mask[j], torch.zeros(attn_mask[j].shape[0], inst_feats[j].shape[0] - attn_mask[j].shape[1], device=attn_mask[j].device).bool()], dim=1)
                if self._box3d_ca_enabled and qpos_ref_list is not None and qsize_ref_norm is not None:
                    queries, _ = self._apply_box3d_ca3d_sp(
                        i, inst_feats, queries, attn_mask,
                        sp_pos_list_elastic=sp_pos_list_elastic,
                        scene_range=scene_range,
                        base_query_pos=qpos_ref_list,
                        ref_sizes_norm=qsize_ref_norm,
                    )
                else:
                    queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
                if use_one2many:
                    one2many_xyz_weights = torch.chunk(super_points[1], len(super_points[0]), dim=0) # torch.chunk(input, chunks, dim=0) 会把输入的张量 input 按照指定的维度 dim 和指定的分块数 chunks 来分割。
                    one2many_attn_mask_score = [scatter_mean(att.float() * xyz_w.view(1, -1), sp, dim=1) # [20000, 1]
                        for att, sp, xyz_w in zip(one2many_attn_mask, super_points[0], one2many_xyz_weights)] # 将点的mask转换为超点的mask
                    one2many_attn_mask = [(att > 0.5).bool() for att in one2many_attn_mask_score] # > 0.5, not <
                    # If attn_mask has all-True row, the result of CA will be nan 所有最小的attn对应的都设定为False
                    for j in range(len(one2many_attn_mask)): # batch_size
                        mask = ~(one2many_attn_mask_score[j] == one2many_attn_mask_score[j].min(dim=1, keepdim=True)[0])
                        one2many_attn_mask[j] *= mask
                    one2many_queries = self.cross_attn_layers[i](inst_feats, one2many_queries, one2many_attn_mask)
            elif self.cross_attn_mode[i+1] == "P" and self.mask_pred_mode[i] == "SP":
                attn_mask = [att[:, sp] for att, sp in zip(attn_mask, super_points[0])]
                queries = self.cross_attn_layers[i](inst_pts_feats, queries, attn_mask)
            elif self.cross_attn_mode[i+1] == "P" and self.mask_pred_mode[i] == "P":
                queries = self.cross_attn_layers[i](inst_pts_feats, queries, attn_mask)
            else:
                raise NotImplementedError("Not support yet!")

            # Optional: Track-window STM injection (after SP cross-attn, before self-attn).
            # This is a no-op unless enabled in config and `track_instances` is provided.
            if bool(getattr(self, "_trk_stm_enabled", False)) and track_instances is not None and query3d_pos is not None:
                queries = self._apply_track_window_stm(queries, track_instances, query3d_pos, layer_idx=i)

            # Optional: DACA-2D injection (paper order = before self-attn)
            if use_daca2d and gdino_cfg.get("order", "paper") == "paper":
                queries = self._apply_daca2d(
                    queries, attn_mask, query2d_feats, query2d_pos, sp_pos_list, gdino_cfg, layer_idx=i
                )

            queries = self.self_attn_layers[i](queries)
            # Optional: DACA-2D injection (code order = after self-attn)
            if use_daca2d and gdino_cfg.get("order", "paper") == "code":
                queries = self._apply_daca2d(
                    queries, attn_mask, query2d_feats, query2d_pos, sp_pos_list, gdino_cfg, layer_idx=i
                )
            queries = self.ffn_layers[i](queries)
            if use_one2many:
                # one2many_queries = self.self_attn_layers[i](one2many_queries)
                one2many_queries = self.ffn_layers[i](one2many_queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            if self.use_decouple and i == 2:
                if self.before_query_memory is not None:
                    queries = self.query_inter([pred_bbox[j][:, :3] for j in range(len(pred_bbox))], queries, self.before_query_memory, self.before_query_memory, self.before_query_boxes)
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask, object_query, pred_bbox = \
                 self._forward_head(queries, mask_feats, mask_pts_feats, last_flag, layer=i+1)
            if use_one2many:
                one2many_cls_pred, one2many_sem_pred, one2many_pred_score, one2many_pred_mask, one2many_attn_mask, one2many_object_query, one2many_pred_bbox = \
                        self._forward_head(one2many_queries, mask_feats, mask_pts_feats, last_flag, layer=i+1)
                # one2many_output = dict(
                #     cls_preds=one2many_cls_pred, sem_preds=one2many_sem_pred, masks=one2many_pred_mask, scores=one2many_pred_score,
                #     queries=one2many_object_query, bboxes=one2many_pred_bbox)
                one2many_output_dict['cls_preds'].append(one2many_cls_pred)
                one2many_output_dict['sem_preds'].append(one2many_sem_pred)
                one2many_output_dict['scores'].append(one2many_pred_score)
                one2many_output_dict['masks'].append(one2many_pred_mask)
                one2many_output_dict['queries'].append(one2many_object_query)
                one2many_output_dict['pred_bboxes'].append(one2many_pred_bbox)
            if self.use_decouple and i == 2:
                self.before_query_memory = [queries[j].clone() for j in range(len(queries))]
                self.before_query_boxes = [pred_bbox[j][:, :3].clone()  for j in range(len(pred_bbox))]
            if self.use_query_memory2 and i in self.query_stage:
                if self.query_memory2[i] is not None:
                    queries_new = self.muti_scale_query[i]([pred_bbox[j][:, :3] for j in range(len(pred_bbox))], queries, self.query_memory2[i], self.query_memory2[i], self.pred_bbox_memory[i])
                    queries = [queries[j] + queries_new[j] for j in range(len(queries))]
                    queries = [self.muti_scale_norm[i](queries[j]) for j in range(len(queries))]
                with torch.no_grad():
                    detach_query = [queries[j].clone().detach() for j in range(len(queries))]
                    detach_pred_bbox = [pred_bbox[j][:, :3].clone().detach() for j in range(len(pred_bbox))]
                self.query_memory2[i] = detach_query
                self.pred_bbox_memory[i] = detach_pred_bbox
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
            object_queries.append(object_query)
            pred_bboxes.append(pred_bbox)

            # Update reference qpos/qsize for box-modulated CA-3D (next layer).
            if self._box3d_ca_enabled and qpos_base_list is not None and qpos_ref_list is not None and qsize_ref_norm is not None:
                try:
                    for b in range(len(queries)):
                        pb = pred_bbox[b] if isinstance(pred_bbox, (list, tuple)) and b < len(pred_bbox) else None
                        if pb is None or (not torch.is_tensor(pb)):
                            continue
                        if b >= len(qpos_base_list) or (not torch.is_tensor(qpos_base_list[b])):
                            continue
                        # Reference centers: base_pos + current predicted offset (avoid accumulation drift).
                        qpos_ref_list[b] = (qpos_base_list[b].to(pb.device, pb.dtype) + pb[:, :3]).detach()
                        # Reference sizes in normalized space (for modulation).
                        if isinstance(scene_range, (list, tuple)) and b < len(scene_range):
                            smin, smax = scene_range[b]
                            if torch.is_tensor(smin) and torch.is_tensor(smax):
                                span = (smax.to(pb.device, pb.dtype) - smin.to(pb.device, pb.dtype)).clamp(min=1e-6)
                                qsize_ref_norm[b] = (pb[:, 3:6] / span).clamp(min=1e-4, max=10.0).detach()
                except Exception:
                    pass

        # Aggregate per-layer DACA-2D apply stats (diagnostics for "is it a no-op?")
        if use_daca2d:
            try:
                self._daca_apply_seen = int(getattr(self, "_daca_apply_seen", 0)) + 1
                layer_stats = getattr(self, "_daca_apply_layer_stats", []) or []
                keys = (
                    "delta_rel_mean",
                    "allowed_q2d_mean",
                    "allowed_q2d_zero_rate",
                    "allowed_q2d_p50",
                    "allowed_q2d_p90",
                    "nq3d_mean",
                    "nq2d_mean",
                    "sp_allow_mean",
                    "sp_allow_p50",
                    "sp_allow_p90",
                    "min_dist_q3d_p50",
                    "min_dist_q3d_p90",
                    "min_dist_q2d_p50",
                    "min_dist_q2d_p90",
                    "q2d_any_sp_rate",
                )
                agg = _mean_over_layers(layer_stats, keys)
                self._last_daca2d_apply_stats = {
                    "seen": int(self._daca_apply_seen),
                    "layers_with_stats": int(len(layer_stats)),
                    "agg": agg,
                    "per_layer": layer_stats,
                }
                if _daca_apply_debug_should_log(gdino_cfg, int(self._daca_apply_seen)):
                    logger = MMLogger.get_current_instance()
                    if logger is not None:
                        logger.info(
                            "[GDINO][daca2d][apply] seen=%d layers=%d delta_rel=%.4g allowed_q2d_mean=%.3f "
                            "allowed_zero=%.3f allowed_p50=%.3f allowed_p90=%.3f nq2d=%.1f nq3d=%.1f "
                            "sp_allow_p50=%.1f min_dist_q3d_p50=%.3f q2d_any_sp=%.3f",
                            int(self._daca_apply_seen),
                            int(len(layer_stats)),
                            float(agg.get("delta_rel_mean", 0.0)),
                            float(agg.get("allowed_q2d_mean", 0.0)),
                            float(agg.get("allowed_q2d_zero_rate", 0.0)),
                            float(agg.get("allowed_q2d_p50", 0.0)),
                            float(agg.get("allowed_q2d_p90", 0.0)),
                            float(agg.get("nq2d_mean", 0.0)),
                            float(agg.get("nq3d_mean", 0.0)),
                            float(agg.get("sp_allow_p50", 0.0)),
                            float(agg.get("min_dist_q3d_p50", 0.0)),
                            float(agg.get("q2d_any_sp_rate", 0.0)),
                        )
            except Exception:
                pass

        # Aggregate per-layer track-window STM apply stats (diagnostics for "is it a no-op?")
        if bool(getattr(self, "_trk_stm_enabled", False)) and bool(getattr(self, "_trk_stm_diag_collect", False)):
            try:
                layer_stats = getattr(self, "_trk_stm_apply_layer_stats", []) or []
                keys = (
                    "gate_alpha",
                    "mem_tracks_total_mean",
                    "mem_tracks_win_mean",
                    "applied_q_mean",
                    "delta_rel_mean",
                    "delta_rel_p50",
                    "delta_rel_p90",
                    "min_dist_p50",
                    "min_dist_p90",
                    "delta_nan",
                )
                agg = _mean_over_layers(layer_stats, keys)
                self._last_trk_stm_apply_stats = {
                    "enabled": True,
                    "mode": str(getattr(self, "_trk_stm_mode", "scale")),
                    "window": int(getattr(self, "_trk_stm_window", 1)),
                    "dist_lambda": float(getattr(self, "_trk_stm_lambda", 1.0)),
                    "gate_init": float(self.track_window_stm_cfg.get("gate_init", -6.0)),
                    "layers_with_stats": int(len(layer_stats)),
                    "agg": agg,
                    "per_layer": layer_stats,
                }
            except Exception:
                self._last_trk_stm_apply_stats = None
        
        neq_sum_list = []
        for cls_pred in cls_preds[:-1]:
            cls_name = cls_pred[0].argmax(-1, keepdim=True).flatten()
            neq_sum = (cls_name != cls_preds[-1][0].argmax(-1, keepdim=True).flatten()).sum(-1)
            neq_sum_list.append(neq_sum.item())
        # print(neq_sum_list)
        aux_outputs = [
            dict(
                cls_preds=cls_pred, sem_preds=sem_pred, masks=masks, scores=scores, bboxes=bboxes)
            for cls_pred, sem_pred, scores, masks, bboxes in zip(
                cls_preds[:-1], sem_preds[:-1], pred_scores[:-1], pred_masks[:-1], pred_bboxes[:-1])]
        if use_one2many:
            one2many_output_aux_outputs = [
                dict(
                    cls_preds=cls_pred, sem_preds=sem_pred, masks=masks, scores=scores, bboxes=bboxes)
                for cls_pred, sem_pred, scores, masks, bboxes in zip(
                    one2many_output_dict['cls_preds'][:-1], one2many_output_dict['sem_preds'][:-1], one2many_output_dict['scores'][:-1], one2many_output_dict['masks'][:-1], one2many_output_dict['pred_bboxes'][:-1])
            ]
            one2many_outputs = dict(
                cls_preds=one2many_output_dict['cls_preds'][-1],
                sem_preds=one2many_output_dict['sem_preds'][-1],
                masks=one2many_output_dict['masks'][-1],
                scores=one2many_output_dict['scores'][-1],
                queries=one2many_output_dict['queries'][-1],
                bboxes=one2many_output_dict['pred_bboxes'][-1],
                aux_outputs=one2many_output_aux_outputs)
            return dict(
                cls_preds=cls_preds[-1],
                sem_preds=sem_preds[-1],
                masks=pred_masks[-1],
                scores=pred_scores[-1],
                queries=object_queries[-1],
                bboxes=pred_bboxes[-1],
                aux_outputs=aux_outputs,
                one2many_outputs=one2many_outputs,)
        else:
            return dict(
                cls_preds=cls_preds[-1],
                sem_preds=sem_preds[-1],
                masks=pred_masks[-1],
                scores=pred_scores[-1],
                queries=object_queries[-1],
                bboxes=pred_bboxes[-1],
                aux_outputs=aux_outputs)
    
    def forward(self, sp_feats, p_feats, queries, super_points, prev_queries=None, use_temporal_loss=False,
                inst_dict=False, track_instances=None, use_one2many=False,
                query2d_feats=None, query2d_pos=None, gdino_daca2d_cfg=None,
                sp_pos_list_override=None,
                sp_pos_list_elastic=None, scene_range=None, query3d_pos=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(
                sp_feats, p_feats, queries, super_points, prev_queries,
                use_temporal_loss=use_temporal_loss, inst_dict=inst_dict,
                track_instances=track_instances, use_one2many=use_one2many,
                query2d_feats=query2d_feats, query2d_pos=query2d_pos,
                gdino_daca2d_cfg=gdino_daca2d_cfg,
                sp_pos_list_override=sp_pos_list_override,
                sp_pos_list_elastic=sp_pos_list_elastic,
                scene_range=scene_range,
                query3d_pos=query3d_pos)
        else:
            raise NotImplementedError("No simple forward!!!")


@MODELS.register_module()
class S3DISQueryDecoder(QueryDecoder):
    # Does it have any differences with QueryDecoder?
    pass


class MergeLayer(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=256, dropout=0.0):
        super(MergeLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 增加LayerNorm来稳定训练
            nn.ReLU(),
            # nn.Dropout(dropout),       # 若需要，使用Dropout提高泛化
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),     # 再次进行LayerNorm
            nn.ReLU(),
            # nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.layer(x)
class FFN2(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt
class QueryInteractionX(nn.Module):
    def __init__(self, in_channels, mid_channels, **kwargs):
        super().__init__()
        dropout = kwargs.get('drop_rate', 0.0)
        self.with_att = kwargs.get('with_att', True)
        self.with_pos = kwargs.get('with_pos', True)
        self.self_attn = nn.MultiheadAttention(in_channels, 8, dropout)
        self.norm1 = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN2(in_channels, mid_channels, dropout)

    def forward(self, track_embed, obj_embed, pos_embed=None):
        track_num = len(track_embed)
        query_embed = torch.cat([track_embed, obj_embed], dim=0)
        
        # add position embedding
        if self.with_pos and pos_embed is not None:
            pos_embed = torch.cat([pos_embed, pos_embed], dim=0)
            q = k = query_embed + pos_embed
        else:
            q = k = query_embed

        tgt = query_embed.clone()
        # attention
        if self.with_att:
            tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm1(tgt)

        # ffn
        tgt = self.ffn(tgt)

        track_embed = tgt[:track_num]
        obj_embed = tgt[track_num:]

        return track_embed, obj_embed
