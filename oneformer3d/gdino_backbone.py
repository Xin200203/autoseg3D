from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


def _ensure_repo_on_path(repo_dir: str) -> None:
    repo_dir = os.path.abspath(os.path.expanduser(repo_dir))
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


@MODELS.register_module()
class GroundingDINOBackbone(BaseModule):
    """Online GroundingDINO feature extractor for AutoSeg3D diagnostics.

    This wrapper is designed to be deterministic and alignment-safe:
    - No hidden random resize/crop inside the model.
    - Caller is expected to provide images already resized to the target size.

    Outputs:
    - srcs: list of multi-level (B,C,H,W) feature maps after input_proj (C=d_model)
    - hs_last: (B,num_queries,d_model) last decoder layer query embeddings
    - pred_boxes: (B,num_queries,4) normalized cxcywh in input image coordinates
    - pred_scores: (B,num_queries) max sigmoid score over text tokens
    """

    def __init__(
        self,
        *,
        repo_dir: str = "/home/nebula/xxy/GroundingDINO",
        config_path: str = "/home/nebula/xxy/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        checkpoint: str = "/home/nebula/xxy/dataset/models/groundingdino_swinb_cogcoor.pth",
        device: str = "cuda",
        caption: str = "object.",
        max_text_len: Optional[int] = None,
        unset_image_tensor: bool = True,
    ) -> None:
        super().__init__()
        self.repo_dir = repo_dir
        self.config_path = config_path
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.caption = caption if caption.endswith(".") else (caption + ".")
        self.max_text_len = max_text_len
        self.unset_image_tensor = bool(unset_image_tensor)

        _ensure_repo_on_path(self.repo_dir)
        from groundingdino.models import build_model  # type: ignore
        from groundingdino.util.misc import clean_state_dict  # type: ignore
        from groundingdino.util.slconfig import SLConfig  # type: ignore

        args = SLConfig.fromfile(self.config_path)
        args.device = str(self.device)
        self.model = build_model(args)
        ckpt = torch.load(self.checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
        self.model.load_state_dict(clean_state_dict(state), strict=False)
        if self.max_text_len is not None:
            try:
                self.model.max_text_len = int(self.max_text_len)
            except Exception:
                pass
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Lazy-import helpers used in the forward replica.
        from groundingdino.models.GroundingDINO.bertwarper import (  # type: ignore
            generate_masks_with_special_tokens_and_transfer_map,
        )

        self._gen_masks = generate_masks_with_special_tokens_and_transfer_map

    @staticmethod
    def _normalize_image(img: torch.Tensor) -> torch.Tensor:
        # img: (B,3,H,W) float in [0,1]
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (img - mean) / std

    def forward(
        self,
        images: torch.Tensor,
        *,
        captions: Optional[List[str]] = None,
        backbone_only: bool = False,
    ) -> Dict[str, Any]:
        """Run GroundingDINO forward and return intermediate features.

        Args:
            images: (B,3,H,W) float tensor in range [0,1], already resized.
            captions: optional list of strings, length B. If None, use default caption.
            backbone_only: if True, only compute `srcs` (multi-level image features)
                and skip text encoder / transformer / box prediction. This is
                much faster and avoids any tokenizer/model downloads.
        """
        _ensure_repo_on_path(self.repo_dir)
        from groundingdino.util.misc import (  # type: ignore
            NestedTensor,
            inverse_sigmoid,
            nested_tensor_from_tensor_list,
        )

        model = self.model
        device = self.device
        images = images.to(device=device)
        if images.dtype != torch.float32:
            images = images.float()
        images = images.clamp(0.0, 1.0)
        images = self._normalize_image(images)

        B = int(images.shape[0])
        backbone_only = bool(backbone_only)

        # --- visual backbone (no text/transformer) ---
        if backbone_only:
            samples = nested_tensor_from_tensor_list(images)
            model.set_image_tensor(samples)
            srcs: List[torch.Tensor] = []
            masks: List[torch.Tensor] = []
            for l, feat in enumerate(model.features):
                src, mask = feat.decompose()
                srcs.append(model.input_proj[l](src))
                masks.append(mask)
                assert mask is not None

            if model.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, model.num_feature_levels):
                    if l == _len_srcs:
                        src = model.input_proj[l](model.features[-1].tensors)
                    else:
                        src = model.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = torch.nn.functional.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    _ = model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)

            if self.unset_image_tensor:
                model.unset_image_tensor()
            return {
                "srcs": srcs,
                "hs_last": None,
                "pred_logits": None,
                "pred_boxes": None,
                "pred_scores": None,
            }

        if captions is None:
            captions = [self.caption] * B
        else:
            captions = [c if c.endswith(".") else (c + ".") for c in captions]

        # --- text dict (copied from GroundingDINO forward) ---
        tokenized = model.tokenizer(captions, padding="longest", return_tensors="pt").to(device)
        (
            text_self_attention_masks,
            position_ids,
            _cate_to_token_mask_list,
        ) = self._gen_masks(tokenized, model.specical_tokens, model.tokenizer)

        if text_self_attention_masks.shape[1] > model.max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, : model.max_text_len, : model.max_text_len]
            position_ids = position_ids[:, : model.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : model.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : model.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : model.max_text_len]

        if model.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = model.bert(**tokenized_for_encoder)
        encoded_text = model.feat_map(bert_output["last_hidden_state"])
        text_token_mask = tokenized.attention_mask.bool()

        if encoded_text.shape[1] > model.max_text_len:
            encoded_text = encoded_text[:, : model.max_text_len, :]
            text_token_mask = text_token_mask[:, : model.max_text_len]
            position_ids = position_ids[:, : model.max_text_len]
            text_self_attention_masks = text_self_attention_masks[:, : model.max_text_len, : model.max_text_len]

        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }

        # --- visual backbone ---
        samples = nested_tensor_from_tensor_list(images)
        model.set_image_tensor(samples)

        srcs: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for l, feat in enumerate(model.features):
            src, mask = feat.decompose()
            srcs.append(model.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, model.num_feature_levels):
                if l == _len_srcs:
                    src = model.input_proj[l](model.features[-1].tensors)
                else:
                    src = model.input_proj[l](srcs[-1])
                m = samples.mask
                mask = torch.nn.functional.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                model.poss.append(pos_l)

        # --- transformer ---
        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = model.transformer(
            srcs, masks, input_query_bbox, model.poss, input_query_label, attn_mask, text_dict
        )

        outputs_coord_list = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], model.bbox_embed, hs):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [layer_cls_embed(layer_hs, text_dict) for layer_cls_embed, layer_hs in zip(model.class_embed, hs)]
        )

        pred_logits = outputs_class[-1]
        pred_boxes = outputs_coord_list[-1]
        pred_scores = pred_logits.sigmoid().max(dim=-1)[0]
        hs_last = hs[-1]

        if self.unset_image_tensor:
            model.unset_image_tensor()

        return {
            "srcs": srcs,
            "hs_last": hs_last,
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "text_dict": text_dict,
        }
