## 第一版：在线引入 SegDINO3D 的 2D 分支（以 GroundingDINO 作为 2D Backbone，先做旁路诊断）

本文件用于 AutoSeg3D 框架内“第一版”接入 SegDINO3D 的 2D 信息（image-level + object-level）时，**最不易错、最可控**的一套实现清单与对照规则。第一版目标是把 **2D→3D 投影对齐**做对，并且在不改变主干策略的前提下输出可诊断的数据（valid_ratio、boxes 数量、query embedding 分布等）。

> 关键背景：ESAM/OneFormer3D 的 DINO 通路已经验证过“在线提取 + 严格 cam_info + 投影 valid_ratio 监控”能跑到 valid≈1.0。第一版优先复用其“对齐约定”，避免在 Resize/Intrinsics/Flip 上重复踩坑。

---

# 0. 第一版范围（必须严格遵守，避免策略污染）

**只做在线提取（不做离线落盘、不改 decoder、不改 merge 策略）**

- 数据 pipeline：新增 `ResizeForGDINO(target_size=固定尺寸)` + `NormalizeCamInfo(strict=True)`（必要时加 `BuildCamInfoFromPoses`）
- 模型：新增 `GroundingDINOBackbone`（返回 `srcs`=image-level feature maps + `hs_last`=object-level query embedding + `pred_boxes/scores`）
- 投影采样：复用 ESAM 的 `projection_utils.project_points_to_uv` + `sample_img_feat`；计算 `valid_ratio`（不达标直接报错/落盘 debug）
- 输出：仅旁路诊断（不影响主干输出/评价指标）

验收标准（第一版必达）：
- `valid_ratio_mean >= 0.99`（subset 上至少稳定接近 1.0）
- 每帧 `n_boxes`、`hs_last` 的范数分布非退化（非全 0 向量）
- 投影失败时：能在日志/输出文件里看到 intrinsics/img_size/pose 等关键信息用于定位

---

# 1. 接口位置规划（AutoSeg3D 内的落点）

## 1.1 数据侧（pipeline）需要补齐的字段

AutoSeg3D 的 MV stage2 数据加载（`LoadAdjacentDataFromFile`）默认不加载图像（`use_FF=False`），第一版需要：

- 在 config 里把 `LoadAdjacentDataFromFile(..., use_FF=True)` 打开，使其产出：
  - `results['img']`: list[ndarray]（逐帧 RGB）
  - `results['img_paths']`: list[str]
  - `results['poses']`: list[4x4]（逐帧 pose/extrinsics；来源于 infos）

并在 pipeline 中新增：

1) `BuildCamInfoFromPoses`（若现有 results 里没有 cam_info/intrinsics）
- 输入：`poses`、（可选）`intrinsics` 或使用 ScanNet 固定 intr
- 输出：`results['cam_info'] = list[dict]`，每帧包含：
  - `intrinsics: Tensor[4] = (fx, fy, cx, cy)`（**对应当前图像坐标系**）
  - `img_size_gdino: Tensor[2] = (H, W)`（**对应当前图像尺寸**）
  - `pose/extrinsics: Tensor[4,4]`
  - `img_valid: bool`
  - （可选）`axis_align_matrix`

2) `ResizeForGDINO(target_size=...)`
- 同步更新：`img` 与 `cam_info[*].intrinsics`、`cam_info[*].img_size_gdino`

3) `NormalizeCamInfo(strict=True)`
- 统一 `cam_info` 的结构，避免 default_collate 产生 list-of-tensor(B) 这种不稳定嵌套

> 参考实现：`3D_Reconstruction/oneformer3d/loading.py:1233`（`ResizeForDINO`）与 `3D_Reconstruction/oneformer3d/loading.py:80`（`NormalizeCamInfo`）。

## 1.2 模型侧（Backbone + 投影）

第一版建议“完全旁路”，不改 decoder，只在 `mixformer3d.extract_feat()` 附近插入：

- `GroundingDINOBackbone.forward(img_tensor_list) -> {srcs, hs_last, pred_boxes, pred_scores}`
- 将 `srcs` 作为 image-level feature maps，对 3D 点做投影采样（得到 point-wise 2D feature，或至少输出 valid_ratio）
- 将 `hs_last/pred_boxes/scores` 仅用于诊断统计（例如 box 数量、score 分布、embedding 范数）

> 注意：GroundingDINO 官方 `util/inference.py` 会做 `RandomResize([800], max_size=1333)`，第一版必须禁用该隐式 resize，保证 resize 只发生在 pipeline 且 cam_info 同步更新。

---

# 2. ResizeForGDINO 对照清单（最容易出错的部分）

本节给出“该怎么 resize + intrinsics 如何更新 + 什么时候 pad/crop/flip”的统一规则。第一版推荐使用 **固定尺寸**（避免 keep_ratio+pad 的复杂性），并保证纵横比一致。

## 2.1 目标尺寸：固定尺寸 vs keep_ratio

第一版推荐（最稳）：
- **固定尺寸（强制）**：例如 `(H,W)=(420,560)` 或 `(480,640)`，并保持与 ScanNet 原始图像 `480x640` 的比例 `H/W=0.75` 一致。
- 这样 `scale_h == scale_w`，不会引入非均匀缩放导致的投影偏差。

不建议第一版使用：
- keep_ratio + letterbox pad（需要额外处理 pad_offset，并在 intrinsics 上加 offset）
- crop（需要更新 principal point，且容易和 3D augmentation/pose 叠加出错）

## 2.2 intrinsics 更新：统一约定与公式

我们把 intrinsics 存成 OpenCV 风格的像素坐标（以像素中心为参考）。在 ESAM 的投影实现里，最终采样是 `grid_sample(..., align_corners=False)`，因此在 **从图像坐标缩放到 feature map 坐标**时会使用 `+0.5/-0.5` 的半像素对齐（见下节 3.2）。

因此在 **ResizeForGDINO（图像域）** 内更新 intrinsics，建议采用“纯 scale”：

- `fx' = fx * (W1 / W0)`
- `fy' = fy * (H1 / H0)`
- `cx' = cx * (W1 / W0)`
- `cy' = cy * (H1 / H0)`

并写入：
- `cam_info[i].intrinsics = (fx',fy',cx',cy')`
- `cam_info[i].img_size_gdino = (H1,W1)`

> 这与 ESAM 的 `ResizeForDINO` 保持一致（见 `3D_Reconstruction/oneformer3d/loading.py:1322-1345`），后续的 `+0.5/-0.5` 统一在“图像→特征图”那一步处理。

## 2.3 如果未来要支持 keep_ratio + pad（第二版再做）

若使用 letterbox（保持比例缩放到 (H',W')，再 pad 到 (H1,W1)），则 intrinsics 更新必须包含 pad 偏移：

1) 先 scale：
- `cx_s = cx * (W' / W0)`
- `cy_s = cy * (H' / H0)`

2) 再加 pad：
- `cx' = cx_s + pad_left`
- `cy' = cy_s + pad_top`

并记录 `pad_left/pad_top` 到 cam_info，以便 debug。

## 2.4 翻转（flip）处理：推荐“UV 镜像”，不直接改 intrinsics

如果 pipeline 中对图像做了水平翻转（或为了数据增强将来要做），最稳的做法是：

- cam_info 里记录 `img_flip=True`（或复用现有 img_meta 的 `flip/img_flip`）
- 在投影得到 `uv` 后，按 feature map 宽度做镜像：
  - `u = (W_feat - 1) - u`

> 参考：`3D_Reconstruction/oneformer3d/mixformer3d.py` 在采样前对 `uv` 做镜像（约 `line 907-913`，具体以版本为准）。

---

# 3. 投影与采样对照清单（valid_ratio 99% 的关键）

## 3.1 坐标系与 pose（最关键的第一性检查）

投影要做对，必须确保：
- `xyz_world` 与 `pose` 在同一世界系（ScanNet 通常是世界系）
- `pose` 是 `world->camera` 还是 `camera->world` 要明确

第一版推荐直接复用 ESAM 的“auto pose pick”（鲁棒模式）：
- 同时尝试：
  - `xyz_cam = xyz_world @ inv(pose)`（world->cam）
  - `xyz_cam = xyz_world @ pose`（若 pose 本身就是 world->cam）
  - （可选）`axis_align @ pose` 的组合
- 选择 `valid_ratio` 最大的模式作为该帧投影方式，并记录统计（`pose_pick_stats`）

> 参考：`3D_Reconstruction/oneformer3d/mixformer3d.py` 中 `_build_dino_fpn_online` 的 `pose_mode='auto'` 分支（会选 valid_ratio 最大的候选）。

## 3.2 “图像 intrinsics → 特征图 intrinsics”的半像素对齐（+0.5/-0.5）

设：
- 输入图像尺寸 `H_img,W_img`（来自 `cam_info.img_size_gdino`）
- 特征图尺寸 `H_feat,W_feat`
- `scale_w = W_feat / W_img`
- `scale_h = H_feat / H_img`

若采样使用 `grid_sample(..., align_corners=False)`，则 ESAM 的对齐规则是：

- `fx_feat = fx_img * scale_w`
- `fy_feat = fy_img * scale_h`
- `cx_feat = (cx_img + 0.5) * scale_w - 0.5`
- `cy_feat = (cy_img + 0.5) * scale_h - 0.5`

随后调用：
- `project_points_to_uv(xyz_cam, feat_hw=(H_feat,W_feat), standard_intrinsics=(fx_feat,fy_feat,cx_feat,cy_feat), already_scaled=True)`
- `sample_img_feat(feat_map, uv, valid, align_corners=False)`

> 参考：`3D_Reconstruction/oneformer3d/mixformer3d.py` 中对 `cx_feat/cy_feat` 的计算，以及 `project_points_to_uv(..., already_scaled=True)`。

## 3.3 valid_ratio 的定义与阈值

- `valid_mask`：投影后同时满足：
  - 深度 `z` 合法（`MIN_DEPTH < z < max_depth`）
  - `0 <= u < W_feat` 且 `0 <= v < H_feat`
- `valid_ratio = valid_mask.float().mean()`

第一版建议：
- `valid_ratio_mean >= 0.99` 视为对齐正确
- 若 `<0.95`：直接抛异常（strict mode），并打印：
  - intrinsics、img_size_gdino、feat_hw、pose_mode、示例 u/v 范围、img_path/lidar_path

---

# 4. GroundingDINOBackbone 输出定义（第一版只要“能对齐 + 可诊断”）

第一版建议 Backbone 返回以下内容（均不改变主干输出）：

## 4.1 image-level：`srcs`

- 多尺度 feature maps（建议取 `input_proj` 后、统一到 `d_model=256` 的那一组）
- 形状示例：`srcs = [B,256,H1,W1], [B,256,H2,W2], ...`

用途：
- 点级特征采样（image-level 2D feature），用于未来 early-fusion / gating

## 4.2 object-level：`hs_last`

- 最后一层 decoder 的 query embedding
- 形状：`hs_last: (B, num_queries, 256)`

用途：
- 未来 DACA-2D：3D queries ↔ 2D queries cross-attention
- 诊断：embedding 范数、分布、是否退化

## 4.3 boxes/scores：`pred_boxes, pred_scores`

- `pred_boxes`：通常是 normalized `cx,cy,w,h`（相对输入图像）
- `pred_scores`：来自 `pred_logits.sigmoid().max(-1)` 或 topk（按你们任务需要）

第一版用途：
- 诊断：每帧 box 数量、score 分布；可选做 “box lift 成 3D center” 的成功率统计

---

# 5. 第一版诊断输出（旁路）

建议在 work_dir 下输出 `gdino_diag/`：

- `gdino_diag_summary.json`
  - `valid_ratio_mean/p50/p95/min`
  - `pose_pick_stats`（inv/direct/identity/axis_align_inv 的比例）
  - `n_boxes_mean/p95`
  - `hs_norm_mean/p05/p95`、`hs_zero_rate`（接近 0 的比例）
  - `src_level_shapes`（各尺度 HxW）
  - （可选）`lift_success_rate`

- `gdino_diag_debug_samples.jsonl`（仅当 strict 失败或 valid_ratio 过低时写入）
  - 每条记录包含：scene_id/frame_id/img_path、intrinsics、img_size、pose_mode、feat_hw、u/v range

---

# 6. 第一版实现顺序（建议）

1) **先让 valid_ratio 跑到 0.99+**
   - 只做：`use_FF=True` + `BuildCamInfoFromPoses` + `ResizeForGDINO` + `NormalizeCamInfo(strict=True)`
   - 模型只输出 `srcs` 并做点级采样 + valid_ratio 统计

2) 再接 `hs_last`、box 数量等旁路统计

3) 等对齐稳定后，才进入第二版：
   - DACA-2D 注入 decoder
   - image-level early-fusion 到 MinkUNet（类似 SegDINO3D 的 `points_2dfeats`）

---

# 7. 记录（changelog）

- 2026-01-17：建立第一版“在线提取 + strict cam_info + valid_ratio 监控”的接入清单（仅旁路诊断，不改主干策略）。

---

# 第二版（V2）：单帧 DACA‑2D（使用 GroundingDINO full forward 的 `hs_last` + distance‑aware mask），先接入 AutoSeg3D Decoder

> 本节是“第二版”的工程设计稿：**面向 AutoSeg3D 的在线 RGB‑D 单帧处理**，不再讨论 SegDINO3D 的多视角 Nearest View Sampling。  
> 因为 AutoSeg3D 的输入是连续帧，但处理逻辑是逐帧（`oneformer3d/mixformer3d.py` 中按 `frame_i` 循环取 `batch_inputs_dict['points'][i][frame_i]`），并且每帧 RGB 与 depth（从而 3D 点）天然一一对应，因此 2D→3D 融合是**单帧内投影/反投影**问题。

## V2 的核心动机（与 V1 的差异）

你明确不做两条“弱版本”：

1) 仅用 `backbone_only` 的 `srcs[level]` 做 ROI pooling 得到 box embedding（你认为意义不大）  
2) 仅做点级 2D→3D 特征拼接/注入（你已有过类似实验，几乎无提升）

因此 V2 直接采用更贴近论文的重路径：

- 使用 GroundingDINO **full forward**，拿到 **object-level queries**：`hs_last`（shape: `[B, Nq, 256]`）
- 同时构造 **distance‑aware mask（DACA‑2D）**，让 3D queries 只 attend 到空间相关的 2D queries
- 在 AutoSeg3D 的 Decoder 中新增一个“可开关”的 **2D query cross-attention 分支**（旁路可先仅记录，不改主干输出）

> 对齐点：SegDINO3D 在 decoder stage 的贡献是“重新注入 2D 语义”（论文 3.1/3.3；代码 `SegDINO3D/segdino3d/models/decoder/instance_seg_3d_decoder.py` 里 `add_dinox_query_ca` 相关段落），而 AutoSeg3D 也已经使用 dot‑product mask head（`torch.einsum('nd,md->nm', ...)`），因此“让 query 表示更语义化”是更符合现状的 ROI 方向。

---

## V2‑A：在 AutoSeg3D Decoder 中接入 DACA‑2D（接口设计）

### A1. 新增配置接口（必须：默认关闭，不影响现有结果）

建议在 `model.test_cfg`（以及后续 `train_cfg`）下增加独立块（名字建议固定，避免漂移）：

```python
test_cfg.gdino_daca2d = dict(
  enable=False,               # 默认 False
  mode="diag_only",           # diag_only | fuse
  # decoder layer 的顺序消融：paper=论文顺序；code=SegDINO3D 开源实现顺序
  order="paper",              # paper | code
  # V2 约束：DACA‑2D **只在 SP 域注入**（不在 P 域注入）
  # 解释：DACA‑2D mask 依赖 `attn_mask_sp: [Nq3d, Nsp]`；P 域是 [Nq3d, Np] 无法直接做 (Nq3d×Nsp)@(Nsp×Nq2d)。
  inject_domain="sp",         # V2 固定为 sp（不开放其它值，避免口径漂移）
  # 注入层（transformer layer 索引 i=0..L-1）：建议 auto_sp
  # - AutoSeg3D stage2 默认 mask_pred_mode=["SP","SP","P","P"]，对应可用的 `attn_mask_sp` 出现在 i=0,1（注意：检查的是 mask_pred_mode[i]）。
  inject_layers="auto_sp",    # auto_sp | [0,1] | []
  frame_stride=1,             # 先做 1（每帧），诊断通过后再考虑 >1
  max_frames=-1,              # -1 不限；可用于 subset 快跑

  # GroundingDINO 模型配置（已本地化）
  backbone=dict(
    type="GroundingDINOBackbone",
    repo_dir="/home/nebula/xxy/GroundingDINO",
    config_path="/home/nebula/xxy/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    checkpoint="/home/nebula/xxy/dataset/models/groundingdino_swinb_cogcoor.pth",
    device="cuda",
    caption="object.",        # 固定 caption；V2 先不要引入复杂文本提示
  ),

  # 2D 输入尺寸（与 cam_info 同步更新；固定尺寸最稳）
  img_size=(420, 560),

  # 2D queries 筛选（full forward 得到 Nq 很大时必须裁剪）
  max_queries=300,
  score_thr=0.05,             # 先很松，避免裁掉有效 query；后续再收紧

  # 2D query 的 3D center 构造（基于当前帧点云投影到 box 的点集）
  query3d_center=dict(
    method="inbox_median",     # inbox_median | inbox_medoid | inbox_mean
    min_support_pts=30,        # 太小的 box 不构 pos（防噪）
  ),

  # DACA-2D 距离门控（distance-aware mask）
  mask=dict(
    enable=True,
    metric="l1",               # 与 SegDINO3D 统一（cdist p=1）
    thr=0.20,                  # 初始值：scene 归一化尺度需要后续调参
    # 可选：每个 3D query 只保留 topk 近邻（更稳但更侵入）
    topk=-1,
  ),

  # 诊断输出
  diag=dict(
    enable=True,
    strict_valid=True,         # 投影 valid_ratio 过低直接报错（沿用 V1）
    out_key="gdino_daca2d",    # 写入 online_monitor.frames[i][out_key]
  )
)
```

> `mode="diag_only"`：只跑 GDINO full forward + query3d_center + mask 构造，写监控，不影响 decoder；  
> `mode="fuse"`：把 `hs_last`（筛选后的）作为 `query2d_feats` 真正注入 decoder 的 cross-attn。

---

### A2. 数据通路（V2 仍然复用 V1 的对齐链路）

V2 不重新设计投影：继续复用 V1 的 `cam_info` 规范化与 `ResizeForGDINO` 的 intrinsics 更新规则。

必要字段（每帧）：
- `img_paths[frame_i]`
- `cam_info[frame_i].intrinsics = (fx,fy,cx,cy)`（对应 resized 后图像域）
- `cam_info[frame_i].pose/extrinsics`
- `cam_info[frame_i].img_size_gdino = (H,W)`（与 GDINO 输入一致）

> V1 已经验证 `valid_ratio_mean≈0.996`（subset64），因此 V2 的第一性风险点不是投影本身，而是 “full forward 的 box/query 如何裁剪 + query3d_center 是否稳定”。

---

### A3. GroundingDINO full forward 输出与裁剪

GroundingDINO full forward 输出（`oneformer3d/gdino_backbone.py` 已支持）：
- `hs_last`: `[B, Nq, 256]`（object-level 2D queries embedding）
- `pred_boxes`: `[B, Nq, 4]`（normalized cxcywh，相对 GDINO 输入图像尺寸）
- `pred_scores`: `[B, Nq]`（query 置信）

V2 必须做裁剪（否则太慢/太大）：
- `keep = (pred_scores >= score_thr)`  
- 若 `keep.sum > max_queries`：按 `pred_scores` topk 取 `max_queries`

并在诊断里记录：
- `nq_raw / nq_keep / score_p50/p90/p95`

---

### A4. query2d_pos（2D object query 的 3D center）如何在单帧内构建

这一步决定 DACA mask 是否有效。单帧下不需要 depth map：直接用当前帧点云 `xyz_world` 投影到 2D box。

步骤（每帧）：
1) 用 `cam_info` 将 `xyz_world -> uv`（图像坐标或特征图坐标均可，但必须与 `pred_boxes` 对齐同一空间）
2) 把 `pred_boxes(cxcywh)` 转为像素 box（xmin,ymin,xmax,ymax），空间与 uv 一致（img_size=420×560）
3) 对每个 box：
   - `in_box = (u in [xmin,xmax] & v in [ymin,ymax] & valid_depth)`
   - `support_pts = xyz_world[in_box]`
   - 若 `len(support_pts) < min_support_pts`：标记该 query 无 pos（丢弃或保留但不参与 mask）
   - 否则 `query2d_pos = median(support_pts)`（推荐 median，抗 outlier）

诊断必须记录：
- `query_pos_valid_rate`（有足够支持点的 query 比例）
- `support_pts_p50/p90`（每个 query 的支持点数）
- `query_pos_range`（xyz 的范围，用于发现坐标系异常）

> 与论文一致性：SegDINO3D 图里说“2D query 的 3D center 是其 mask 投影点的 medoid”；我们这里 box-only 是近似，但单帧 RGB‑D 下 box 与 depth 同源，且 DACA mask 只需要 coarse 空间约束，通常足够作为 V2 起步。

---

### A5. DACA‑2D distance‑aware mask（在 AutoSeg3D Decoder 的实现接口）

AutoSeg3D decoder 当前 forward 入口（训练与推理都共享）：
- `ScanNetMixQueryDecoder.forward_iter_pred(sp_feats, p_feats, queries, super_points, ...)`  

V2 需要为 decoder 增加两个可选参数（默认 None，不影响现有调用）：
- `query2d_feats: Optional[List[Tensor]]`（每 batch 一个 `[Nq_keep, 256]`）
- `query2d_pos: Optional[List[Tensor]]`（每 batch 一个 `[Nq_keep, 3]`，用于 mask）

在每层 decoder 中，V2 的目标顺序是（与论文描述对齐）：

顺序做消融（两种都支持，默认按论文）：

**Order=`paper`（论文）**：`3D cross-attn → DACA‑2D → 3D self-attn → FFN`  
**Order=`code`（SegDINO3D 开源实现）**：`3D cross-attn → 3D self-attn → DACA‑2D → FFN`

> 校准说明：SegDINO3D 代码在 `SegDINO3D/segdino3d/models/decoder/instance_seg_3d_decoder.py` 中实际为 `cross → self → DACA → ffn`，但论文文字描述更接近 `cross → DACA → self → ffn`。V2 在 AutoSeg3D 中两种顺序都保留，**用相同配置/子集跑消融**，以数据裁决。

#### 先对齐 AutoSeg3D 的真实执行顺序（避免“论文顺序”写对但代码插错）

AutoSeg3D 当前的 `ScanNetMixQueryDecoder.forward_iter_pred()` 内，每个 transformer layer 的固定顺序是：

- `3D cross-attn` → `3D self-attn` → `FFN` → `_forward_head`（更新 `attn_mask` 给下一层用）

因此 V2 里 `order` 的含义要明确落到“插入点”：

- `order="paper"`：把 DACA‑2D 插在 **cross-attn 之后、self-attn 之前**
- `order="code"`：把 DACA‑2D 插在 **self-attn 之后、FFN 之前**（与 SegDINO3D 开源实现一致）

这两种插入点都只改变 query 表示，不会改变现有 `_forward_head` 产出的 mask 维度；因此适合作为纯顺序消融。

#### distance-aware mask 的“正确做法”（关键校准点）

SegDINO3D 的 DACA‑2D mask 不是简单的 `dist<thr`，而是利用“当前 3D query 的空间支持区域”（由 `attn_mask` 指向的 superpoints）去选择可见的 2D queries：

- 已有 `attn_mask_sp`: `[Nq3d, Nsp]`（bool，True=禁止 attend；这是你们每层由 `pred_mask` 产生的 mask-attn）
- 先构造 superpoint 的 3D 位置 `sp_pos`（AutoSeg3D 可以在 decoder 内用点坐标与 sp_id 计算）：
  - `xyz = p_feats[:, :3]`（点的世界坐标；AutoSeg3D 的 `p_feats` 本身是 `torch.cat([xyz, feat], dim=-1)`）
  - `sp_id = super_points[0]`（点到 superpoint 的映射）
  - `sp_pos = scatter_mean(xyz, sp_id, dim=0)` → `[Nsp, 3]`
- 对 2D object query 的 3D center（`query2d_pos: [Nq2d, 3]`）计算距离：
  - `dist = cdist(sp_pos, query2d_pos, p=1)` → `[Nsp, Nq2d]`
- 把“3D query 可 attend 的 superpoints”映射成“3D query 可 attend 的 2D queries”：
  - `reach = (~attn_mask_sp).float() @ (dist < thr).float()` → `[Nq3d, Nq2d]`
  - `daca_attn_mask = (reach == 0)`（bool，True=禁止 attend）

这一步能显著降低“mask 全开/全关”的不稳定性：每个 3D query 只会去 attend 与其当前 spatial support 接近的 2D queries。

#### 重要限制（必须在实现与配置里显式控制）

上述公式要求 `attn_mask` 的 key 维度是 superpoints（`Nsp`）。在 AutoSeg3D 的默认 stage2 配置里：
- `mask_pred_mode=["SP","SP","P","P"]`
因此 V2 **明确只在 SP 域层插入 DACA‑2D**，不在 P 域注入（避免维度不一致与额外映射带来的工程风险）：

- `inject_domain="sp"`（固定）
- `inject_layers="auto_sp"`（推荐）：实现上按当前层 `mask_pred_mode[i]=="SP"` 自动启用（对应 i=0,1）；也可显式指定 `[0,1]`。

> 备注：若未来要在 P 域层也注入，需要额外把点域 `attn_mask` 映射回 SP 域或改用 query positional state；这不属于 V2 范围。

#### 防 NaN 的 dummy query（与 SegDINO3D 对齐）

对齐 SegDINO3D 的策略：为避免出现某个 3D query 对所有 2D queries 都被 mask（导致 attention 数值不稳定），追加一个 dummy 2D query：
- `query2d_feats <- cat([query2d_feats, ones(1,256)], dim=0)`
- `daca_attn_mask <- cat([daca_attn_mask, zeros(Nq3d,1)], dim=-1)`（放开最后一列）

诊断（每帧）记录：
- `mask_open_ratio`（mask 为 False 的比例，或每个 query 可见 2D query 数分位数）
- `empty_mask_rows_rate`（需要 dummy 的比例）

---

## V2‑B：实验建议（按最小风险逐步打开）

1) `mode="diag_only"`：只跑 full GDINO + query3d_center + mask 统计  
   - 验收：`valid_ratio` 仍≈1；`query_pos_valid_rate` 不太低（否则 DACA mask 无意义）；`empty_mask_rows_rate` 不应接近 1

2) `mode="fuse"` + `mask.enable=True`：真正把 `hs_last` 注入 decoder  
   - 首轮不要再做其它结构改动（不引入 image-level 2D→3D 注入；不引入额外 NMS/merge），保证单变量。

> 你要求“必须同时用 hs_last + DACA mask 才可能有提升”，因此 V2 的 fuse 版本默认 `mask.enable=True`，不提供“无 mask 的 hs_last 注入”作为主线（但可作为 ablation 留接口）。

---

## V2 关键风险与应对（必须提前写死）

1) **full GDINO 计算成本过高**
   - 先在 subset64 做；必要时 frame_stride>1 或 max_frames 限制；最终再考虑 cache（但 V2 首版不做离线 cache）

2) **query3d_center 支持点太少**
   - 单帧点云稀疏时 box 内点可能少；需要 `min_support_pts` 可配，并记录分布；必要时扩大 box（可配 `box_expand_px`）

3) **mask 太严格导致 attention 全空**
   - 必须记录 `empty_mask_rows_rate`，并实现 dummy query 保底

4) **坐标系不一致**
   - 继续沿用 V1 的 pose_mode auto pick（但单帧 RGB‑D 下通常固定为某一 mode），一旦 valid_ratio 掉就 strict 报错

---

# V2 记录（changelog）

- 2026-01-17：V2 设计稿：面向 AutoSeg3D 的在线单帧 RGB‑D，采用 GDINO full forward 的 `hs_last` + DACA‑2D distance-aware mask 注入 decoder（不再讨论多视角 Nearest View Sampling）。
