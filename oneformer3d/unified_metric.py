import torch
import numpy as np
import os
import json
from typing import Optional, Sequence, Dict
from mmengine.logging import MMLogger

from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.registry import METRICS
from mmdet3d.evaluation import panoptic_seg_eval, seg_eval
from .instance_seg_eval import instance_seg_eval, instance_cat_agnostic_eval

@METRICS.register_module()
class UnifiedSegMetric(SegMetric):
    """Metric for instance, semantic, and panoptic evaluation.
    The order of classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    def __init__(self,
                 thing_class_inds,
                 stuff_class_inds,
                 min_num_points,
                 id_offset,
                 sem_mapping,   
                 inst_mapping,
                 metric_meta,
                 logger_keys=[('miou',),
                              ('all_ap', 'all_ap_50%', 'all_ap_25%'), 
                              ('pq',)],
                 online_monitor: Optional[dict] = None,
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.metric_meta = metric_meta
        # self.logger_keys = logger_keys
        self.logger_keys = [('all_ap', 'all_ap_50%', 'all_ap_25%')]
        self.sem_mapping = np.array(sem_mapping)
        self.inst_mapping = np.array(inst_mapping)
        self.online_monitor = online_monitor or {}
        super().__init__(**kwargs)

    @staticmethod
    def _resolve_out_dir(logger: MMLogger, out_dir: str) -> str:
        base = None
        # Prefer work dir; log_dir may point to repo root in some setups.
        for attr in ("work_dir", "log_dir", "output_dir"):
            base = getattr(logger, attr, None)
            if base:
                break
        # Fallback: infer from file handler path (usually under work_dir/timestamp).
        if not base:
            try:
                for h in getattr(logger, "handlers", []) or []:
                    fn = getattr(h, "baseFilename", None)
                    if fn:
                        base = os.path.dirname(str(fn))
                        break
            except Exception:
                base = None
        if not base:
            base = os.getcwd()
        if os.path.isabs(out_dir):
            return out_dir
        return os.path.join(str(base), out_dir)

    @staticmethod
    def _summarize_online_monitor(monitors: Sequence[dict]) -> dict:
        """Lightweight summary over per-scene online_monitor dicts."""
        def _collect(path: str) -> np.ndarray:
            # path like "assoc.matched" or "sp_merge.batch.0.merge_drop"
            keys = path.split(".")
            vals = []
            for s in monitors:
                for fr in s.get("frames", []) if isinstance(s.get("frames", None), list) else []:
                    cur = fr
                    ok = True
                    for k in keys:
                        if isinstance(cur, dict):
                            if k in cur:
                                cur = cur[k]
                                continue
                            ok = False
                            break
                        if isinstance(cur, list) and k.isdigit():
                            idx = int(k)
                            if 0 <= idx < len(cur):
                                cur = cur[idx]
                                continue
                            ok = False
                            break
                        ok = False
                        break
                    if ok and isinstance(cur, (int, float, np.number)):
                        vals.append(float(cur))
            return np.asarray(vals, dtype=np.float32)

        def _collect_scene(path: str) -> np.ndarray:
            keys = path.split(".")
            vals = []
            for s in monitors:
                cur = s
                ok = True
                for k in keys:
                    if isinstance(cur, dict):
                        if k in cur:
                            cur = cur[k]
                            continue
                        ok = False
                        break
                    if isinstance(cur, list) and k.isdigit():
                        idx = int(k)
                        if 0 <= idx < len(cur):
                            cur = cur[idx]
                            continue
                        ok = False
                        break
                    ok = False
                    break
                if ok and isinstance(cur, (int, float, np.number)):
                    vals.append(float(cur))
            return np.asarray(vals, dtype=np.float32)

        def _pack(x: np.ndarray) -> dict:
            if x.size == 0:
                return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0}
            return {
                "n": int(x.size),
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "p90": float(np.percentile(x, 90)),
                "p95": float(np.percentile(x, 95)),
            }

        det = _collect("assoc.det")
        matched = _collect("assoc.matched")
        birth = _collect("assoc.birth")
        revived = _collect("assoc.revived")
        track_after = _collect("assoc.track_valid_after")
        sp_merge_drop = _collect("sp_merge.batch.0.merge_drop")
        det_after_nms = _collect("det_filter.after_nms")
        det_after_inst = _collect("det_filter.after_inst_thr")
        det_after_np = _collect("det_filter.after_npoint_thr")
        scene_ngt = _collect_scene("scene_gt_dup.n_gt")
        scene_npred = _collect_scene("scene_gt_dup.n_pred")
        scene_hit0_05 = _collect_scene("scene_gt_dup.iou05.hit0")
        scene_hit_ge2_05 = _collect_scene("scene_gt_dup.iou05.hit_ge2")
        scene_hit0_01 = _collect_scene("scene_gt_dup.iou01.hit0")
        scene_hit_ge2_01 = _collect_scene("scene_gt_dup.iou01.hit_ge2")
        scene_best_01_05 = _collect_scene("scene_gt_dup.best_iou.btw0p1_0p5")
        scene_mult01_ge2 = _collect_scene("scene_gt_dup.iou01.hit_ge2")

        assoc_birth_mean_scene = _collect_scene("scene_assoc_summary.birth_mean")
        assoc_birth_sum_scene = _collect_scene("scene_assoc_summary.birth_sum")
        assoc_track_after_mean_scene = _collect_scene("scene_assoc_summary.track_after_mean")
        inflation_scene = _collect_scene("scene_assoc_summary.inflation")

        return {
            "counts": {
                "scenes": int(len(monitors)),
                "frames": int(sum(len(s.get("frames", [])) for s in monitors if isinstance(s.get("frames", None), list))),
            },
            "assoc": {
                "det": _pack(det),
                "matched": _pack(matched),
                "birth": _pack(birth),
                "revived": _pack(revived),
                "track_valid_after": _pack(track_after),
            },
            "det_filter": {
                "after_nms": _pack(det_after_nms),
                "after_inst_thr": _pack(det_after_inst),
                "after_npoint_thr": _pack(det_after_np),
            },
            "sp_merge": {
                "merge_drop": _pack(sp_merge_drop),
            },
            "scene_gt_dup": {
                "n_gt": _pack(scene_ngt),
                "n_pred": _pack(scene_npred),
                "iou05": {
                    "hit0": _pack(scene_hit0_05),
                    "hit_ge2": _pack(scene_hit_ge2_05),
                },
                "iou01": {
                    "hit0": _pack(scene_hit0_01),
                    "hit_ge2": _pack(scene_hit_ge2_01),
                },
                "best_iou": {
                    "btw0p1_0p5": _pack(scene_best_01_05),
                },
            },
            "derived": {
                "corr_dup01_ge2_birth_mean": float(np.corrcoef(scene_mult01_ge2, assoc_birth_mean_scene)[0, 1]) if scene_mult01_ge2.size >= 2 and assoc_birth_mean_scene.size == scene_mult01_ge2.size else 0.0,
                "corr_dup01_ge2_birth_sum": float(np.corrcoef(scene_mult01_ge2, assoc_birth_sum_scene)[0, 1]) if scene_mult01_ge2.size >= 2 and assoc_birth_sum_scene.size == scene_mult01_ge2.size else 0.0,
                "corr_dup01_ge2_inflation": float(np.corrcoef(scene_mult01_ge2, inflation_scene)[0, 1]) if scene_mult01_ge2.size >= 2 and inflation_scene.size == scene_mult01_ge2.size else 0.0,
                "corr_dup01_ge2_track_after_mean": float(np.corrcoef(scene_mult01_ge2, assoc_track_after_mean_scene)[0, 1]) if scene_mult01_ge2.size >= 2 and assoc_track_after_mean_scene.size == scene_mult01_ge2.size else 0.0,
            },
        }

    @staticmethod
    def _compute_scene_gt_dup(
        gt_inst: torch.Tensor,
        pred_inst: torch.Tensor,
        min_gt_points: int = 100,
        iou_thr: float = 0.5,
        iou_lo_thr: float = 0.1,
    ) -> Optional[Dict]:
        """Scene-level GT-view multiplicity statistics.

        For each GT instance (>= min_gt_points), count how many predicted
        instances overlap it (IoU>=thr). This directly measures duplicate
        tracks: GT hit_ge2 means multiple tracks hit the same GT.
        """
        if gt_inst is None or pred_inst is None:
            return None
        if gt_inst.numel() == 0 or pred_inst.numel() == 0:
            return None
        # GT is always per-point ids; pred may be either per-point ids or [P, N] boolean masks.
        gt = gt_inst.detach().to("cpu", dtype=torch.long).flatten()

        if pred_inst.dim() == 2:
            pred_masks = pred_inst.detach().to("cpu").bool()
            if pred_masks.shape[1] != gt.shape[0]:
                return None
            pred_sizes = pred_masks.sum(1).to(dtype=torch.float32)
            keep_pred = pred_sizes > 0
            pred_masks = pred_masks[keep_pred]
            pred_sizes = pred_sizes[keep_pred]
            P = int(pred_masks.shape[0])
        else:
            pr = pred_inst.detach().to("cpu", dtype=torch.long).flatten()
            if pr.shape != gt.shape:
                return None
            pr_valid = pr >= 0
            if int(pr_valid.sum().item()) == 0:
                pred_masks = torch.zeros((0, gt.shape[0]), dtype=torch.bool)
                pred_sizes = torch.zeros((0,), dtype=torch.float32)
                P = 0
            else:
                pr_ids, pr_inv = torch.unique(pr[pr_valid], return_inverse=True)
                P = int(pr_ids.numel())
                pred_masks = torch.zeros((P, gt.shape[0]), dtype=torch.bool)
                pred_masks[:, pr_valid] = torch.nn.functional.one_hot(pr_inv, num_classes=P).t().bool()
                pred_sizes = pred_masks.sum(1).to(dtype=torch.float32)

        gt_valid = gt >= 0
        if int(gt_valid.sum().item()) == 0:
            return {"n_gt": 0, "n_pred": int(P),
                    "iou05": {"hit0": 0.0, "hit_ge2": 0.0, "mean_mult": 0.0},
                    "iou01": {"hit0": 0.0, "hit_ge2": 0.0, "mean_mult": 0.0}}

        gt_ids, gt_inv = torch.unique(gt[gt_valid], return_inverse=True)
        gt_cnt = torch.bincount(gt_inv, minlength=int(gt_ids.numel()))
        vis_g = gt_cnt >= int(min_gt_points)
        if int(vis_g.sum().item()) == 0:
            return {"n_gt": 0, "n_pred": int(P),
                    "iou05": {"hit0": 0.0, "hit_ge2": 0.0, "mean_mult": 0.0},
                    "iou01": {"hit0": 0.0, "hit_ge2": 0.0, "mean_mult": 0.0}}

        # Build visible GT masks [G, N]
        gt_remap = torch.full((int(gt_ids.numel()),), -1, dtype=torch.long)
        gt_remap[vis_g] = torch.arange(int(vis_g.sum().item()), dtype=torch.long)
        gt_idx_full = torch.full_like(gt, -1)
        gt_idx_full[gt_valid] = gt_remap[gt_inv]
        keep_pts = gt_idx_full >= 0
        G = int(vis_g.sum().item())

        if G == 0:
            return {"n_gt": 0, "n_pred": int(P),
                    "iou05": {"hit0": 0.0, "hit_ge2": 0.0, "mean_mult": 0.0},
                    "iou01": {"hit0": 0.0, "hit_ge2": 0.0, "mean_mult": 0.0}}

        # GT sizes for visible GTs
        gt_sizes = torch.bincount(gt_idx_full[keep_pts], minlength=G).to(dtype=torch.float32)  # [G]

        # IoU matrix [P, G] via per-pred bincount over visible GT indices (much cheaper than dense matmul).
        if P == 0:
            return {
                "n_gt": int(G),
                "n_pred": 0,
                "iou05": {"hit0": 1.0, "hit_ge2": 0.0, "mean_mult": 0.0, "hit1": 0.0, "hit_ge3": 0.0},
                "iou01": {"hit0": 1.0, "hit_ge2": 0.0, "mean_mult": 0.0, "hit1": 0.0, "hit_ge3": 0.0},
                "best_iou": {"eq0": 1.0, "lt0p1": 0.0, "btw0p1_0p5": 0.0, "ge0p5": 0.0},
                "mult_iou01": {"hit0": 1.0, "hit1": 0.0, "hit2": 0.0, "hit3p": 0.0},
            }

        iou = torch.zeros((P, G), dtype=torch.float32)
        inter = torch.zeros((P, G), dtype=torch.float32)
        for p_idx in range(P):
            m = pred_masks[p_idx]
            if m.numel() != gt.numel():
                continue
            sel = m & keep_pts
            if int(sel.sum().item()) == 0:
                continue
            gsel = gt_idx_full[sel]
            inter_row = torch.bincount(gsel, minlength=G).to(dtype=torch.float32)
            inter[p_idx] = inter_row
        ps = pred_sizes.unsqueeze(1)  # [P,1]
        gs = gt_sizes.unsqueeze(0)    # [1,G]
        union = ps + gs - inter
        iou = inter / (union + 1e-6)

        def _hit(iou_mat: torch.Tensor, thr: float) -> Dict[str, float]:
            hit = (iou_mat >= thr).sum(0)  # [G]
            hit0 = float((hit == 0).float().mean().item()) if hit.numel() else 0.0
            hit1 = float((hit == 1).float().mean().item()) if hit.numel() else 0.0
            hit_ge2 = float((hit >= 2).float().mean().item()) if hit.numel() else 0.0
            hit_ge3 = float((hit >= 3).float().mean().item()) if hit.numel() else 0.0
            pos = hit[hit >= 1].to(dtype=torch.float32)
            mean_mult = float(pos.mean().item()) if pos.numel() else 0.0
            return {"hit0": hit0, "hit1": hit1, "hit_ge2": hit_ge2, "hit_ge3": hit_ge3, "mean_mult": mean_mult}

        # Best IoU distribution (per GT), useful to understand gray-zone (0.1~0.5).
        best_iou = iou.max(0).values if iou.numel() else torch.zeros((G,), dtype=torch.float32)
        best_eq0 = float((best_iou == 0).float().mean().item()) if best_iou.numel() else 0.0
        best_lt01 = float(((best_iou > 0) & (best_iou < 0.1)).float().mean().item()) if best_iou.numel() else 0.0
        best_01_05 = float(((best_iou >= 0.1) & (best_iou < 0.5)).float().mean().item()) if best_iou.numel() else 0.0
        best_ge05 = float((best_iou >= 0.5).float().mean().item()) if best_iou.numel() else 0.0

        # Multiplicity histogram at IoU>=0.1 (per GT)
        hit01 = (iou >= float(iou_lo_thr)).sum(0) if iou.numel() else torch.zeros((G,), dtype=torch.long)
        mult_hit0 = float((hit01 == 0).float().mean().item()) if hit01.numel() else 0.0
        mult_hit1 = float((hit01 == 1).float().mean().item()) if hit01.numel() else 0.0
        mult_hit2 = float((hit01 == 2).float().mean().item()) if hit01.numel() else 0.0
        mult_hit3p = float((hit01 >= 3).float().mean().item()) if hit01.numel() else 0.0

        return {
            "n_gt": int(G),
            "n_pred": int(P),
            "iou05": _hit(iou, float(iou_thr)),
            "iou01": _hit(iou, float(iou_lo_thr)),
            "best_iou": {"eq0": best_eq0, "lt0p1": best_lt01, "btw0p1_0p5": best_01_05, "ge0p5": best_ge05},
            "mult_iou01": {"hit0": mult_hit0, "hit1": mult_hit1, "hit2": mult_hit2, "hit3p": mult_hit3p},
        }

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        
        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']
        label2cat = self.metric_meta['label2cat']
        ignore_index = self.metric_meta['ignore_index']
        classes = self.metric_meta['classes']
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)

        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []
        online_monitor_results = []
        bbox_center_diag_results = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        mon_cfg = self.online_monitor or {}
        scene_cfg = mon_cfg.get("scene_gt_dup", {}) if isinstance(mon_cfg, dict) else {}
        scene_enable = True
        if isinstance(scene_cfg, dict) and ("enable" in scene_cfg):
            scene_enable = bool(scene_cfg.get("enable", False))
        scene_min_pts = int(scene_cfg.get("min_gt_points", 100)) if isinstance(scene_cfg, dict) else 100
        scene_iou_thr = float(scene_cfg.get("iou_thr", 0.5)) if isinstance(scene_cfg, dict) else 0.5
        scene_iou_lo = float(scene_cfg.get("iou_lo_thr", 0.1)) if isinstance(scene_cfg, dict) else 0.1

        for eval_ann, single_pred_results in results:

            sem_mask, inst_mask = self.map_inst_markup(
                eval_ann['pts_semantic_mask'].copy(), 
                eval_ann['pts_instance_mask'].copy(), 
                self.valid_class_ids[num_stuff_cls:],
                num_stuff_cls)

            gt_semantic_masks_inst_task.append(sem_mask)
            gt_instance_masks_inst_task.append(inst_mask)           
            pred_instance_masks_inst_task.append(
                torch.tensor(single_pred_results['pts_instance_mask'][0]))
            pred_instance_labels.append(
                torch.tensor(single_pred_results['instance_labels']))
            pred_instance_scores.append(
                torch.tensor(single_pred_results['instance_scores']))
            if 'bbox_center_diag' in single_pred_results and isinstance(single_pred_results['bbox_center_diag'], dict):
                bbox_center_diag_results.append(single_pred_results['bbox_center_diag'])
            if 'online_monitor' in single_pred_results and isinstance(single_pred_results['online_monitor'], dict):
                mon = single_pred_results['online_monitor']
                # Scene-level GT-view duplicate stats (side-channel; no effect on eval).
                if bool(mon_cfg.get("enable", False)) and scene_enable:
                    try:
                        dup = self._compute_scene_gt_dup(
                            gt_inst=torch.as_tensor(inst_mask),
                            pred_inst=torch.as_tensor(single_pred_results['pts_instance_mask'][0]),
                            min_gt_points=scene_min_pts,
                            iou_thr=scene_iou_thr,
                            iou_lo_thr=scene_iou_lo,
                        )
                        if dup is not None:
                            mon["scene_gt_dup"] = dup
                    except Exception:
                        pass

                # Scene-level association summary (birth vs inflation signals).
                try:
                    births = []
                    track_after = []
                    dets = []
                    matcheds = []
                    for fr in mon.get("frames", []) if isinstance(mon.get("frames", None), list) else []:
                        a = fr.get("assoc", {}) if isinstance(fr, dict) else {}
                        if isinstance(a, dict):
                            if "birth" in a:
                                births.append(float(a["birth"]))
                            if "track_valid_after" in a:
                                track_after.append(float(a["track_valid_after"]))
                            if "det" in a:
                                dets.append(float(a["det"]))
                            if "matched" in a:
                                matcheds.append(float(a["matched"]))
                    birth_sum = float(np.sum(births)) if births else 0.0
                    birth_mean = float(np.mean(births)) if births else 0.0
                    track_after_mean = float(np.mean(track_after)) if track_after else 0.0
                    det_mean = float(np.mean(dets)) if dets else 0.0
                    matched_mean = float(np.mean(matcheds)) if matcheds else 0.0
                    inflation = 0.0
                    if isinstance(mon.get("scene_gt_dup", None), dict):
                        ng = float(mon["scene_gt_dup"].get("n_gt", 0))
                        npred = float(mon["scene_gt_dup"].get("n_pred", 0))
                        inflation = float(npred / ng) if ng > 0 else 0.0
                    mon["scene_assoc_summary"] = {
                        "birth_sum": birth_sum,
                        "birth_mean": birth_mean,
                        "track_after_mean": track_after_mean,
                        "det_mean": det_mean,
                        "matched_mean": matched_mean,
                        "inflation": inflation,
                    }
                except Exception:
                    pass
                online_monitor_results.append(mon)

        if pred_instance_labels[0].max() == 0:
            ret_inst = instance_cat_agnostic_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[num_stuff_cls:],
                class_labels=classes[num_stuff_cls:-1],
                logger=logger)
        else:
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[num_stuff_cls:],
                class_labels=classes[num_stuff_cls:-1],
                logger=logger)

        metrics = dict()
        # for ret, keys in zip((ret_sem, ret_inst, ret_pan), self.logger_keys):
        for ret, keys in zip((ret_inst,), self.logger_keys):
            for key in keys:
                metrics[key] = ret[key]

        # Optional: dump online monitor (raw + summary) under work_dir.
        if bool(mon_cfg.get("enable", False)) and len(online_monitor_results) > 0:
            out_dir = str(mon_cfg.get("out_dir", "online_monitor"))
            out_dir = self._resolve_out_dir(logger, out_dir)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "online_monitor.json"), "w", encoding="utf-8") as f:
                json.dump(online_monitor_results, f, indent=2, ensure_ascii=False, default=str)
            summary = self._summarize_online_monitor(online_monitor_results)
            with open(os.path.join(out_dir, "online_monitor_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"[UnifiedSegMetric] online monitor saved to: {out_dir}")

            # Optional: dump bbox/center(+feat) diag payloads (npz + summary) under the same out_dir.
            diag_cfg = mon_cfg.get("bbox_center_diag", {}) if isinstance(mon_cfg, dict) else {}
            if bool(diag_cfg.get("enable", False)) and len(bbox_center_diag_results) > 0:
                try:
                    def _cat(key: str) -> np.ndarray:
                        xs = []
                        for d in bbox_center_diag_results:
                            arr = d.get(key, None)
                            if isinstance(arr, np.ndarray):
                                xs.append(arr)
                            elif isinstance(arr, (list, tuple)):
                                xs.append(np.asarray(arr, dtype=np.float32))
                        if len(xs) == 0:
                            cols = 2 + (1 if bool(diag_cfg.get("with_feat", True)) else 0) + (1 if bool(diag_cfg.get("with_voxel", False)) else 0)
                            return np.zeros((0, cols), dtype=np.float32)
                        return np.concatenate(xs, axis=0).astype(np.float32, copy=False)

                    dt_pos = _cat("det_track_pos")
                    dt_neg = _cat("det_track_neg")
                    tt_pos = _cat("track_track_pos")
                    tt_neg = _cat("track_track_neg")
                    np.savez_compressed(
                        os.path.join(out_dir, "bbox_center_diag_all.npz"),
                        det_track_pos=dt_pos,
                        det_track_neg=dt_neg,
                        track_track_pos=tt_pos,
                        track_track_neg=tt_neg,
                    )

                    # Provide a compact threshold grid for quick selection.
                    iou_thrs = diag_cfg.get("bbox_iou_thrs", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                    cen_thrs = diag_cfg.get("center_norm_thrs", [0.1, 0.2, 0.3, 0.5, 1.0])
                    iou_thrs = [float(x) for x in iou_thrs]
                    cen_thrs = [float(x) for x in cen_thrs]

                    dt_iou_curve = [{"thr": t, "pos": float(np.mean(dt_pos[:, 0] >= t)) if dt_pos.size else 0.0,
                                     "neg": float(np.mean(dt_neg[:, 0] >= t)) if dt_neg.size else 0.0}
                                    for t in iou_thrs]
                    tt_iou_curve = [{"thr": t, "pos": float(np.mean(tt_pos[:, 0] >= t)) if tt_pos.size else 0.0,
                                     "neg": float(np.mean(tt_neg[:, 0] >= t)) if tt_neg.size else 0.0}
                                    for t in iou_thrs]
                    dt_cen_curve = [{"thr": t, "pos": float(np.mean(dt_pos[:, 1] <= t)) if dt_pos.size else 0.0,
                                     "neg": float(np.mean(dt_neg[:, 1] <= t)) if dt_neg.size else 0.0}
                                    for t in cen_thrs]
                    tt_cen_curve = [{"thr": t, "pos": float(np.mean(tt_pos[:, 1] <= t)) if tt_pos.size else 0.0,
                                     "neg": float(np.mean(tt_neg[:, 1] <= t)) if tt_neg.size else 0.0}
                                    for t in cen_thrs]

                    feat_enable = bool(diag_cfg.get("with_feat", True)) and (dt_pos.shape[1] >= 3)
                    vox_enable = bool(diag_cfg.get("with_voxel", False)) and (dt_pos.shape[1] >= (4 if feat_enable else 3))
                    feat_thrs = diag_cfg.get("feat_thrs", [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9])
                    feat_thrs = [float(x) for x in feat_thrs]
                    vox_thrs = diag_cfg.get("vox_thrs", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                    vox_thrs = [float(x) for x in vox_thrs]
                    dt_feat_curve = None
                    tt_feat_curve = None
                    if feat_enable:
                        dt_feat_curve = [{"thr": t, "pos": float(np.mean(dt_pos[:, 2] >= t)) if dt_pos.size else 0.0,
                                          "neg": float(np.mean(dt_neg[:, 2] >= t)) if dt_neg.size else 0.0}
                                         for t in feat_thrs]
                        tt_feat_curve = [{"thr": t, "pos": float(np.mean(tt_pos[:, 2] >= t)) if tt_pos.size else 0.0,
                                          "neg": float(np.mean(tt_neg[:, 2] >= t)) if tt_neg.size else 0.0}
                                         for t in feat_thrs]

                    dt_vox_curve = None
                    tt_vox_curve = None
                    vox_idx = 3 if feat_enable else 2
                    if vox_enable:
                        dt_vox_curve = [{"thr": t, "pos": float(np.mean(dt_pos[:, vox_idx] >= t)) if dt_pos.size else 0.0,
                                         "neg": float(np.mean(dt_neg[:, vox_idx] >= t)) if dt_neg.size else 0.0}
                                        for t in vox_thrs]
                        tt_vox_curve = [{"thr": t, "pos": float(np.mean(tt_pos[:, vox_idx] >= t)) if tt_pos.size else 0.0,
                                         "neg": float(np.mean(tt_neg[:, vox_idx] >= t)) if tt_neg.size else 0.0}
                                        for t in vox_thrs]

                    def _gate(arr: np.ndarray, iou_thr: float, cen_thr: float) -> float:
                        if arr.size == 0:
                            return 0.0
                        return float(np.mean((arr[:, 0] >= iou_thr) & (arr[:, 1] <= cen_thr)))

                    comb = []
                    for a in iou_thrs:
                        for b in cen_thrs:
                            comb.append({
                                "iou": a,
                                "center": b,
                                "det_track_pos": _gate(dt_pos, a, b),
                                "det_track_neg": _gate(dt_neg, a, b),
                                "track_track_pos": _gate(tt_pos, a, b),
                                "track_track_neg": _gate(tt_neg, a, b),
                            })

                    comb3 = None
                    if feat_enable:
                        comb3 = []
                        for a in iou_thrs:
                            for b in cen_thrs:
                                for c in feat_thrs:
                                    def _gate3(arr: np.ndarray) -> float:
                                        if arr.size == 0:
                                            return 0.0
                                        return float(np.mean((arr[:, 0] >= a) & (arr[:, 1] <= b) & (arr[:, 2] >= c)))
                                    comb3.append({
                                        "iou": a,
                                        "center": b,
                                        "feat": c,
                                        "det_track_pos": _gate3(dt_pos),
                                        "det_track_neg": _gate3(dt_neg),
                                        "track_track_pos": _gate3(tt_pos),
                                        "track_track_neg": _gate3(tt_neg),
                                    })

                    comb4 = None
                    if feat_enable and vox_enable:
                        comb4 = []
                        for a in iou_thrs:
                            for b in cen_thrs:
                                for c in feat_thrs:
                                    for v in vox_thrs:
                                        def _gate4(arr: np.ndarray) -> float:
                                            if arr.size == 0:
                                                return 0.0
                                            return float(np.mean((arr[:, 0] >= a) & (arr[:, 1] <= b) & (arr[:, 2] >= c) & (arr[:, 3] >= v)))
                                        comb4.append({
                                            "iou": a,
                                            "center": b,
                                            "feat": c,
                                            "vox": v,
                                            "det_track_pos": _gate4(dt_pos),
                                            "det_track_neg": _gate4(dt_neg),
                                            "track_track_pos": _gate4(tt_pos),
                                            "track_track_neg": _gate4(tt_neg),
                                        })

                    diag_summary = {
                        "counts": {
                            "det_track_pos": int(dt_pos.shape[0]),
                            "det_track_neg": int(dt_neg.shape[0]),
                            "track_track_pos": int(tt_pos.shape[0]),
                            "track_track_neg": int(tt_neg.shape[0]),
                        },
                        "det_track": {"bbox_iou_curve": dt_iou_curve, "center_norm_curve": dt_cen_curve},
                        "track_track": {"bbox_iou_curve": tt_iou_curve, "center_norm_curve": tt_cen_curve},
                        "combined": comb,
                        "feat": {
                            "enable": bool(feat_enable),
                            "det_track_curve": dt_feat_curve,
                            "track_track_curve": tt_feat_curve,
                        },
                        "voxel": {
                            "enable": bool(vox_enable),
                            "det_track_curve": dt_vox_curve,
                            "track_track_curve": tt_vox_curve,
                        },
                        "combined3": comb3,
                        "combined4": comb4,
                    }
                    with open(os.path.join(out_dir, "bbox_center_diag_summary.json"), "w", encoding="utf-8") as f:
                        json.dump(diag_summary, f, indent=2, ensure_ascii=False)
                    logger.info(f"[UnifiedSegMetric] bbox/center diag saved to: {out_dir}")
                except Exception as e:
                    logger.warning(f"[UnifiedSegMetric] bbox/center diag dump failed: {repr(e)}")
        return metrics

    def map_inst_markup(self,
                        pts_semantic_mask,
                        pts_instance_mask,
                        valid_class_ids,
                        num_stuff_cls):
        """Map gt instance and semantic classes back from panoptic annotations.

        Args:
            pts_semantic_mask (np.array): of shape (n_raw_points,)
            pts_instance_mask (np.array): of shape (n_raw_points.)
            valid_class_ids (Tuple): of len n_instance_classes
            num_stuff_cls (int): number of stuff classes
        
        Returns:
            Tuple:
                np.array: pts_semantic_mask of shape (n_raw_points,)
                np.array: pts_instance_mask of shape (n_raw_points,)
        """
        pts_instance_mask -= num_stuff_cls
        pts_instance_mask[pts_instance_mask < 0] = -1
        
        pts_semantic_mask -= num_stuff_cls
        pts_semantic_mask[pts_instance_mask == -1] = -1

        mapping = np.array(list(valid_class_ids) + [-1])
        pts_semantic_mask = mapping[pts_semantic_mask]
        
        return pts_semantic_mask, pts_instance_mask
