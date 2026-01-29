## 第一版：在线引入 SegDINO3D 的 2D 分支（以 GroundingDINO 作为 2D Backbone，先做旁路诊断）

本文件用于 AutoSeg3D 框架内“第一版”接入 SegDINO3D 的 2D 信息（image-level + object-level）时，**最不易错、最可控**的一套实现清单与对照规则。第一版目标是把 **2D→3D 投影对齐**做对，并且在不改变主干策略的前提下输出可诊断的数据（valid_ratio、boxes 数量、query embedding 分布等）。

> 关键背景：ESAM/OneFormer3D 的 DINO 通路已经验证过“在线提取 + 严格 cam_info + 投影 valid_ratio 监控”能跑到 valid≈1.0。第一版优先复用其“对齐约定”，避免在 Resize/Intrinsics/Flip 上重复踩坑。

---

# -1. 本轮“校准结论”：我们到底要引入 SegDINO3D 的什么

你当前的目标不是“在 AutoSeg3D 上堆一些 2D 特征试试”，而是**尽可能严格地把 SegDINO3D 的 2D×3D 联合机制迁移过来**。两者共同祖先都是 OneFormer3D，因此正确做法是：

- **先对齐结构（结构对齐优先于调参）**：UNet early‑fusion（point‑level 2D→3D） + decoder object‑level DACA‑2D（2D object queries → 3D queries）
- **再对齐坐标空间（这是最容易“看似启用但完全无效”的坑）**：DACA 距离门控必须与 decoder 的 3D 空间一致；投影 box/support 必须与相机几何一致
- **最后对齐损失/匹配**：SegDINO3D 的 box center/size 监督并不是“数据集提供 bbox”，而是训练时从 GT mask+点坐标算出来的

本文件后续把“已实现/未实现/需要修复”的颗粒度写到**函数/张量维度/空间约定**级别，避免“改了一堆但效果等价于没改”。

---

# -2. SegDINO3D vs AutoSeg3D：对照总览（到文件/张量级别）

## -2.1 SegDINO3D 的关键工程点（你要迁移的“核心三件事”）

1) **Point‑level early fusion（送进 MinkUNet）**  
SegDINO3D 不是只做 decoder 注入，它把每个 3D 点对应的 2D feature（256d）拼到点云输入里：  
- `in_channels = 256 + 3`（3D 的 RGB/颜色 3 通道 + 2D point feature 256 通道）  
- backbone 输出 `out_channels=96`  
代码参考：`SegDINO3D/configs/models/base_3d.py` 与 `SegDINO3D/segdino3d/models/backbone/minkunet.py`（cat RGB + points_2dfeats）

2) **Object‑level DACA‑2D（2D object queries 注入 decoder）**  
SegDINO3D 的 DACA‑2D 注入不是 ROI pooling，而是：  
- 使用 2D foundation model 的 object queries embedding（`dinox_queries`，维度 256）  
- 每个 2D query 还需要一个 3D anchor（`dinox_query_pos`，维度 3）  
- DACA mask 用 `pos_wo_elastic (Nsp,3)` 与 `dinox_query_pos (Nq2d,3)` 做 `cdist`，再由 `attn_mask (Nq3d,Nsp)` 映射到 `Nq2d`  
代码参考：`SegDINO3D/segdino3d/models/decoder/instance_seg_3d_decoder.py` 中 `add_dinox_query_ca/add_dinox_query_ca_mask`

3) **Box center/size 监督不是“数据集 bbox”**  
SegDINO3D 在训练时从 GT instance mask + 点坐标直接计算：  
- `instance_centers (n_inst,3)`  
- `instance_sizes (n_inst,3)`  
并在 matcher/loss 里启用 CenterL1Cost、SizeL1Cost 与对应 L1 loss。  
代码参考：`SegDINO3D/segdino3d/models/architecture/baseline3d.py:get_extra_instance_data()` 与 `SegDINO3D/segdino3d/models/loss/loss_3d.py`

> 结论：如果只做 “GDINO backbone_only 的 srcs + ROI pooling” 或 “仅 point‑feature 拼接”，都无法完整复现 SegDINO3D 的收益路径；你要求的 V2 必须同时具备 early fusion + object‑level DACA‑2D 才符合论文/代码主线。

## -2.2 AutoSeg3D 当前分支“已实现/部分实现/未实现”

### 已实现（代码已存在，可复用）
- `AutoSeg3D/oneformer3d/gdino_backbone.py`：`GroundingDINOBackbone` 支持 `backbone_only` 与 `full forward`，能输出 `srcs / hs_last / pred_boxes / pred_scores`
- `AutoSeg3D/oneformer3d/loading.py`：存在 `ResizeForGDINO`、`NormalizeCamInfo`、`BuildCamInfoFromPoses`（严格规范化 cam_info 是对齐关键）
- `AutoSeg3D/oneformer3d/mixformer3d.py`：  
  - `_run_gdino_point_fusion_for_frame()`：对每帧投影采样 `srcs[level]` 形成 point‑wise 2D feature，并带 `valid_ratio` 监控  
  - `_run_gdino_daca2d_for_frame()`：full forward 后构造 `query2d_feats/query2d_pos`（基于 box 内 support points 的 3D median）
- `AutoSeg3D/oneformer3d/query_decoder.py`：  
  - 已实现 “SP 域专用”的 DACA‑2D 注入 `_apply_daca2d()`  
  - 支持 `order="paper"/"code"` 的顺序消融（见 `gdino_daca2d_cfg.order`）

### 维度对照（帮助你快速排查“维度不一致导致无效注入”）

以 SegDINO3D 的 base 配置为参照（`d_model=256, backbone out_channels=96`），两边在“decoder 主干维度”上是天然对齐的：

- **2D object queries embedding**：`hs_last / dinox_queries` → `256d`
- **decoder d_model**：`256`
- **3D backbone 输出（送入 decoder 的 in_channels）**：`96`
- **decoder input_proj**：`96 → 256`
- **mask head（dot product）**：query(256) 与 mask_feats(256) 做相似度（两边都属于 OneFormer3D 系列常见实现）

因此：V2 里 object‑level 的注入“理论上不需要额外投影层”，只要 `hs_last` 经过筛选后仍是 `[...,256]`，即可直接作为 2D query embedding 进入 decoder 的 cross‑attn。

### 目前仍缺/仍存在偏差（这些会让 V2 训练“看似启用但无效”）
- **训练路径未保证调用 object‑level DACA‑2D**：当前 DACA 的数据/开关主要挂在 `test_cfg.gdino_daca2d`，需要明确 `loss()`/训练 forward 是否也会构造并传入 `query2d_feats/query2d_pos`
- **DACA 距离门控空间一致性未被“写死保证”**：SegDINO3D 明确用 `pos_wo_elastic`（非 elastic 的 SP 位置）做距离；AutoSeg3D 当前 DACA 使用 `sp_pos_list`（来源于 decoder 内的点坐标），必须明确它对应的是哪套坐标（aug? elastic?）
- **early fusion 维度未对齐 SegDINO3D**：SegDINO3D 是 `points_2dfeats=256d` 直接拼入 UNet；而 AutoSeg3D 当前 SV 路线 A 配置（`AutoSeg3D/configs/scannet200/AutoSeg3D_sv_scannet200_routeA_gdino.py`）是 `gdino_in_dim=256 → gdino_out_dim=32`，UNet 输入因此是 `3+32`，这更像“轻量注入”而不是 SegDINO3D‑style early fusion（需要在配置里提供 256d 对齐开关）
- **loss/matcher 未对齐 SegDINO3D 的 center/size 监督**：AutoSeg3D 当前 InstanceCriterion 与 bbox_flag 配置（center/size L1、box modulation）仍不完整；这会削弱 query/geometry 对齐能力
- **现有 `_run_gdino_daca2d_for_frame()` 尚未完全遵守 raw_proj_space → decoder_space_wo_elastic 的约定**：当前实现直接用 `batch_inputs_dict['points']` 投影并选 support points，但 point_fusion 已经表明“训练时必须 reverse aug 才能保证与 pose/intr 对齐”；因此 DACA‑2D 的 support 选点也必须复用同样的 reverse‑aug 投影逻辑，否则会出现 “valid_ratio 看似不低但 box‑support 语义漂移” 的隐蔽失真。

---

# -3. 最关键的第一性约束：DACA‑2D 的三套空间必须写清并强制一致

SegDINO3D 的经验是：**DACA‑2D 很容易“启用但无效”，原因不是网络，而是空间错位**。在 AutoSeg3D 里必须把三套空间明确成硬约定：

## -3.1 三套空间定义（建议命名）

- `raw_proj_space`：用于 2D 投影 / in_box 选 support points 的空间  
  - 必须与相机 `pose/intr` 一致（不受 3D aug 影响）  
  - 对应 AutoSeg3D 已在 point_fusion 做的：`apply_3d_transformation(reverse=True)` 后再投影

- `decoder_space_wo_elastic`：decoder 里做“距离门控”的空间（DACA mask 的距离阈值就在这个空间里）  
  - 必须与 decoder 内 `sp_pos_list` 的坐标同一套（同一 augmentation，但**不含 elastic**）  
  - SegDINO3D 用 `pos_wo_elastic` 明确做到这一点

- `decoder_space_elastic`：如果启用了 elastic deformation（仅用于特征学习），它不应进入 DACA 的几何距离

## -3.2 直接结论（必须按此实现，否则 DACA 无意义）

- **in_box/support points 的判定必须在 raw_proj_space 做**（否则 box-support 语义错位）
- **query2d_pos 必须写入 decoder_space_wo_elastic**（否则 DACA 距离阈值错位）
- **DACA 距离计算必须使用 sp_pos_wo_elastic（而不是 elastic 后的位置）**

> 这也是你之前在 ESAM 的 point_fusion 链路能达到 valid≈1 的根因：投影与相机几何一致；把同样原则复制到 DACA‑2D 的 q_pos 上，才能让 DACA 真正“触发”。

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

> 本节是“第二版”的工程设计稿：**面向 AutoSeg3D 的在线 RGB‑D 单帧处理**。  
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

#### 为什么需要“box 内 support points”，不能只靠 2D box 数学计算？

`pred_boxes(cxcywh)` 本质是 **2D 图像平面上的区域**。它并不携带深度，因此**无法唯一确定一个 3D 位置**：同一个 2D box 对应一条视锥（frustum）。  
在 RGB‑D/点云场景里，3D anchor 的“正解”只能来自深度（或点云）：

- 你可以“用 depth map 在 box 内做统计”得到一个 3D 点（等价于 support points）
- 你也可以“用点云投影到 box 内筛 support points”，再用 median/mean/medoid 得到 anchor（工程上更直接，且与后续 3D 点/超点索引天然对齐）

因此：support points 不是为了“更复杂”，而是为了把 2D box **lift 成一个 3D anchor（query2d_pos）**，让 DACA 的距离门控在 3D 空间有意义。

#### 对齐风险点（非常关键）：raw 投影 + aug 写入（并且 DACA 距离在 wo_elastic 空间）

训练时 AutoSeg3D 存在 3D aug（rot/scale/trans/flip + 可能的 elastic）。相机参数（pose/intr）对应的是**原始相机几何**，因此：

- **in_box 判定必须用 raw_proj_space 的点去投影**（即把增强后的点先 reverse 回相机一致的空间再投影）
- 但 **query2d_pos 必须写入 decoder_space_wo_elastic**（与 decoder 内 DACA 距离门控空间一致）

最稳的实现约定（与你们已验证的 point_fusion 经验一致）：

1) 用 `xyz_aug`（当前 batch 的点坐标）通过 `apply_3d_transformation(reverse=True)` 得到 `xyz_raw_for_proj`  
2) 用 `xyz_raw_for_proj` + `pose/intr` 投影得到 uv，并做 in_box/support selection（raw_proj_space）  
3) 对同一批点的索引集合 `in_box_idx`，回到 `xyz_aug_wo_elastic`（或“与 sp_pos_list 一致的坐标”）上取这些点的坐标，计算 median 得 `query2d_pos`（decoder_space_wo_elastic）  
4) decoder 内 DACA mask 的距离必须用 `sp_pos_wo_elastic` 与 `query2d_pos` 计算（不要用 elastic 后的位置）

> SegDINO3D 代码里显式使用 `pos_wo_elastic` 做 DACA 距离：`dist = cdist(pos_wo_elastic, dinox_query_pos, p=1)`。  
> AutoSeg3D 若不引入 “sp_pos_wo_elastic” 的显式接口，就必须保证 `sp_pos_list` 本身就是 wo_elastic 空间，否则 DACA 阈值与距离将失去物理意义。

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
- `order="code"`：把 DACA‑2D 插在 **self-attn 之后、FFN 之前**（与 SegDINO3D 开源代码的执行顺序一致）

> 重要校准：论文中描述的顺序是 `3D cross-attn → DACA‑2D → 3D self-attn → FFN`；  
> 但 SegDINO3D 开源实现里实际是 `3D cross-attn → 3D self-attn → DACA‑2D → FFN`（见 `SegDINO3D/segdino3d/models/decoder/instance_seg_3d_decoder.py: forward_iter_pred`）。  
> 因此 AutoSeg3D 这里同时保留 paper/code 两种插入点是合理的：用实验消融裁决即可。

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

## V2‑C：SegDINO3D‑style 的 UNet early‑fusion（256d points_2dfeats，送入 MinkUNet）

这部分是你最终要“完整复现 SegDINO3D 路径”时必须补齐的第二条腿（与 DACA‑2D 并列）。仅靠 decoder 注入通常不足以带来稳定增益。

SegDINO3D 的结构约束（必须对齐）：
- UNet 输入：`[rgb(3), points_2dfeats(256)]`（等价 `in_channels = 256 + 3`；xyz 坐标不作为特征通道，而是 MinkowskiEngine 的坐标）
- UNet 输出：`out_channels=96`，再进入 decoder 的 `input_proj (96→256)`

AutoSeg3D 当前已有的基础设施：
- `_run_gdino_point_fusion_for_frame()` 已能对每帧：`project → sample srcs[level] → 得到 per-point 2D feat` 并监控 `valid_ratio`

尚需明确并对齐的关键点（必须写进后续实现计划）：
- **points_2dfeats 维度**：需要支持输出 256d（而不是压缩到 32d）作为可选配置，才能与 SegDINO3D 可比
- **特征层选择**：SegDINO3D 的 256d 对应其 2D backbone 的 d_model；GDINO 的 `srcs` 也是 256d，理论上可以直接使用（避免再 MLP 投影导致信息瓶颈）
- **拼接位置**：必须发生在 MinkUNet 第一个卷积之前（即真正的 early fusion），而不是 decoder 里 late fusion

建议 V2.1 的最小可跑版本：
- 先只取单尺度 `srcs[level=0]`（最密）→ point_fusion 得 `points_2dfeats_256`
- 仅做拼接进入 UNet（不做任何新损失），验证训练/推理链路正确

---

## V2‑D：与 SegDINO3D 对齐的 “box center/size” 监督与 matcher（后续阶段）

这不是“现在就要改”的部分，但必须提前把事实写清楚，避免继续误解成“SV 数据缺 bboxes_3d”：

- SegDINO3D 的 `instance_centers/instance_sizes` 是**运行时从 GT mask + 点坐标计算出来**，不是数据集字段
- 因此 AutoSeg3D 若要对齐，需要在训练 forward（loss 之前）补一个与 `Baseline3D.get_extra_instance_data()` 等价的步骤：
  - 输入：GT instance masks（n_inst,n_pts）+ 点坐标（优先用 `pos_wo_elastic`）
  - 输出：`instance_centers/instance_sizes` 写入 targets/data_samples
  - matcher 加 `CenterL1Cost/SizeL1Cost`，loss 加对应 L1 项

此外 SegDINO3D 的 decoder 还有：
- `box_modulate_ca`（cross-attn 的 box size modulation）
- `normalize_box_prediction`
这些属于“结构增强”，建议在 V2.2 之后再考虑（先把 early fusion + DACA‑2D 跑稳）。

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

- 2026-01-17：V2 设计稿：面向 AutoSeg3D 的在线单帧 RGB‑D，采用 GDINO full forward 的 `hs_last` + DACA‑2D distance-aware mask 注入 decoder。
- 2026-01-21：补充 V3 对齐清单：offline SV 验证、DACA‑2D mask L1 对齐、decoder 深度对齐（6 层 ablation）、center/size 监督的运行时计算与 matcher/loss 对齐。

---

# 第三版（V3）：SV 单帧训练“一次到位”对齐 SegDINO3D（early fusion + object-level DACA‑2D + box center/size loss）

你当前明确要求：**第一轮训练就以“完整更改后的模型形态”训练**（UNet / decoder / loss 都按 SegDINO3D 对齐），中间只允许做最小自测（投影 valid_ratio、维度一致性、loss 不 NaN）。

本节给出“可直接落地改代码”的清晰计划：每个模块的输入输出维度、需要新增/改动的接口、以及必须对齐的超参开关。

> 重要说明：AutoSeg3D 与 SegDINO3D 的 3D backbone 同属 Res16UNet34 系列（OneFormer3D lineage）。  
> 它们的“每一层结构（planes/blocks/stride）”可以保持一致；真正决定差异的是：**输入通道数、2D 特征是否 early-fusion、decoder 是否启用 box positional embedding 与 DACA‑2D**。  
> 因此 V3 的目标不是“改出一个新 UNet”，而是“把 UNet 的输入/融合方式、decoder 的注入与 loss 监督”对齐到 SegDINO3D。

## V3‑0：总开关（SV 单帧训练）

### 目标运行形态（强制）
- 数据：SV（每样本 `T=1`），RGB 与 depth/点云天然一一对应（**无需最近视角采样**）
- 2D backbone：GroundingDINO（在线提取），必须 full forward 得 `hs_last/pred_boxes/pred_scores`
- 3D backbone：Res16UNet34C（MinkowskiEngine）
- decoder：保留原有 3D CA 主干，同时增加 DACA‑2D（**仅 SP 域注入**）
- loss：matcher + loss 同时加入 center/size 监督（与 SegDINO3D 一致）

### 非常重要：SV 训练必须先在 “offline detector” 上对齐验证
AutoSeg3D 的 Online 类（`ScanNet200MixFormer3D_Online`）会引入 `memory/merge` 等在线机制，即便你把 `use_query_memory=False`，仍可能在 forward/predict 路径里走到不同的分支，给“2D 注入是否有效”的判断带来干扰。

因此 V3 的**对齐验证顺序**必须是：
1) **offline SV**：`ScanNet200MixFormer3D`（或对应的 offline detector），`T=1`，只验证“2D 注入 + loss”是否按 SegDINO3D 生效  
2) 通过后再迁移到 Online（Stage2 / online reconstruction），否则会把“tracking/memory 的系统行为”与“2D 注入是否有效”混在一起

### V3 对齐清单（最小但关键）
以下 4 条是“最小但关键”的对齐项，优先级从高到低（任何一条缺失都会显著削弱 SegDINO3D 的收益路径）：
1) **DACA‑2D mask 距离度量改为 L1（p=1）**：`mask.metric='l1'`，`thr=0.2` 保持不变  
   - SegDINO3D 是 `torch.cdist(..., p=1)`；若用 L2@0.2 会更严格，导致可 attend 的 2D query 更少，注入更弱。
2) **decoder 深度对齐到 6 层（至少做 ablation）**：`num_layers: 3 → 6`  
   - 先验证“注入深度不足导致 2D 信息无法传递”这一假设；再决定是否进一步对齐 decoder 结构（box‑modulated CA 等）。
3) **加入 center/size 监督（matcher cost + loss）**：实现 SegDINO3D 的 `CenterL1Cost/SizeL1Cost` 与对应 L1 loss  
   - 注意：center/size GT 是运行时从 GT mask + 点坐标算出来的（不是数据集提供 bbox）。
4) **保持原 3D CA 主干不被覆盖**：DACA‑2D 作为 SP 域的增量注入模块  
   - 不应删/替换原有 3D cross‑attn/self‑attn；DACA‑2D 是“额外一条信息通路”，不是主干替代。

### V3‑0.1 坐标系校准清单（V3 规范段落 / 验收标准，必须长期遵守）

> 这部分是 V3 里**最容易“看起来启用但实际无效/甚至变差”**的根因：同一个样本在训练中同时存在多套 3D 坐标（raw / aug / elastic）。  
> **正确做法不是“全程只用一个坐标系”，而是：每个模块必须使用它约定的坐标空间，并且三套空间之间要“索引一致、定义清晰、传参明确”。**

#### 0) AutoSeg3D 真实存在的三套 3D 坐标（必须统一命名）

- **(A) raw（投影空间）**：`points_raw[:, :3]`  
  - 来源：pipeline 的 `SavePointsForProjection`（保存 3D aug 之前的点）。  
  - 目的：与相机几何 `cam_info.pose/intrinsics` 严格一致，用于 `uv/valid/in_box` 这类 2D-3D 投影判定。
- **(B) aug/wo-elastic（decoder 距离门控空间）**：`points[:, :3]`  
  - 含义：经过 `RandomFlip3D + GlobalRotScaleTrans`（rigid 3D aug）之后的点，**不包含 elastic**（elastic 只写 `elastic_coords`，不改 `points`）。  
  - 目的：用于 DACA‑2D 的 3D anchor（`query2d_pos`）写入，以及 DACA mask 的距离门控（与 decoder 内部 `sp_pos_list` 一致）。
- **(C) elastic（3D 特征/box 空间）**：`elastic_coords * voxel_size`  
  - 含义：elastic augmentation 后的点坐标（Minkowski sparse conv 的主特征空间）。  
  - 目的：用于 3D box‑modulated CA‑3D（SegDINO3D Sec 3.3）与 box loss 的监督空间；也用于 `scene_range` 的归一化。

#### 1) 各模块必须使用的坐标空间（写死的工程约束）

1) **GDINO point‑fusion（image‑level）**  
   - `uv/valid` 投影必须使用 **raw：`points_raw`**（相机几何只在 raw 空间成立）。  
   - 抽样成功率监控：`valid_ratio` 应接近 1（建议 `>=0.95` strict）。

2) **GDINO DACA‑2D 的 support points 选择（box 内点）**  
   - `in_box` 判定必须使用 **raw 投影**得到的 `uv`（来自 `points_raw`）。  
   - 这是“这个 2D box 覆盖到哪些 3D 点”的唯一正确依据。

3) **GDINO DACA‑2D 的 3D anchor 写入（`query2d_pos`）**  
   - `query2d_pos` 必须写在 **aug/wo-elastic：`points`** 空间：对 `in_box` 的点索引 `idx`，用 `points[idx].median/mean` 得到 `q_pos`。  
   - 目的：保证 DACA‑2D 的距离门控在 decoder 的 3D 空间里可用（与 `sp_pos_list` 同空间）。

4) **DACA‑2D 距离门控（mask）**  
   - `dist = cdist(sp_pos_wo_elastic, query2d_pos, p=1)` 时，`sp_pos_wo_elastic` 必须来自 **aug/wo-elastic：`points`** 的 superpoint scatter mean；`query2d_pos` 同样来自 `points`。  
   - 注意：这里的 “wo‑elastic” 指的是**不带 elastic**；不是 `points_raw`。  
   - 违反该规则会出现一种非常隐蔽的现象：`valid_ratio` 仍很高，但 `allowed_zero`（3D queries 没有任何 2D query 可 attend）会显著升高，最终等价于注入 no-op。

5) **3D box‑modulated CA‑3D（SegDINO3D Sec 3.3）与 box loss（L_box）**  
   - **必须在 elastic 空间**：GT 的 `bboxes_3d / instance_centers / instance_sizes` 与 `scene_range` 都要用 `elastic_coords*voxel_size` 计算（若无 elastic，则退化到 `points[:,:3]`）。  
   - 预测的 `pred_centers/pred_sizes` 同样在 elastic 空间解释。  
   - 目的：避免 “loss 在 elastic、attention 在 points” 或反之导致监督失真。

#### 2) 强制自检（只要不满足就应该直接报错/退出训练）

每个 batch（SV，T=1）至少检查：
- **索引一致性**：`points_raw.shape[0] == points.shape[0] == num_sample`；若有 `elastic_coords` 则 `elastic_coords.shape[0] == points.shape[0]`。  
  - 不一致代表 `SavePointsForProjection` 放置错误或 pipeline 在保存 raw 后又重采样/裁剪了点（投影会整体错位）。
- **投影只走 raw**：任何使用 `points`/`elastic_coords` 做相机投影都应视为 bug。  
- **DACA 距离空间一致**：`query2d_pos` 与 `sp_pos_wo_elastic` 必须来自同一套 `points`（aug/wo-elastic）。  
- **训练期监控**（建议长期打印/汇总）：  
  - `point_fusion.valid_ratio_mean/min`  
  - `daca2d.valid_ratio_mean/min` + `qpos_rate_mean/min` + `nq_keep_mean/nq_pos_mean`  
  - `daca2d.apply.allowed_zero`（如果持续很高，说明门控空间或阈值有问题）

## V3‑1：Point‑level early fusion（严格对齐 SegDINO3D：256d 直接进 UNet）

### 目标（对齐事实）

SegDINO3D 的 early fusion 是：
- `points_2dfeats ∈ R^{N×256}` 直接拼接到点特征里（不做 256→32 压缩）
- UNet `in_channels = 256 + 3`（3 是 RGB；xyz 是坐标，不是 feature channel）

### AutoSeg3D 当前差异（需要修）
- 现有 SV routeA 配置是 `gdino_in_dim=256 → gdino_out_dim=32`，UNet 输入为 `3+32`（轻量注入）

### V3 实施方案（接口级）

1) `gdino_point_fusion.out_dim=256`（与 `srcs[level].C=256` 对齐）
2) 让 `gdino_point_fusion` 输出的每点特征 **不再额外降维**：
   - 最稳妥：把 `self._gdino_point_proj` 设为 identity（或 256→256 的 Linear 但默认可不启用）
3) UNet 输入通道改为：
   - `backbone.in_channels = 3 + 256 = 259`
4) 训练增强：
   - 引入与 SegDINO3D 一致的 `dropout_rate_2dfeats`（例如 0.7）在训练时随机 dropout `points_2dfeats`（只对 2D 部分做 dropout，不动 RGB）

### 关键 I/O（落到张量维度）

- 输入点特征：`feat_in = concat([rgb(3), gdino_point_feats(256)]) -> (N,259)`
- UNet 输出：`point_feat_3d = (N,96)`（与现有 decoder input_proj 对齐）

> 备注：这会导致 UNet 第一层卷积权重 shape 改变，因此不能直接加载旧 backbone 权重；你已经接受从头训练，这点是预期行为。

## V3‑2：Object‑level DACA‑2D（严格对齐 SegDINO3D：pos_wo_elastic 做距离门控）

### 目标（对齐事实）

SegDINO3D 的 DACA‑2D（见 `SegDINO3D/segdino3d/models/decoder/instance_seg_3d_decoder.py`）要求：
- `dinox_queries (Nq2d,256)`：2D object queries embedding
- `dinox_query_pos (Nq2d,3)`：每个 2D query 的 3D anchor
- 距离门控用 `pos_wo_elastic (Nsp,3)` 与 `dinox_query_pos (Nq2d,3)` 做 `cdist`

### AutoSeg3D 当前状态（可复用但需“空间对齐修复”）

已存在：
- `_run_gdino_daca2d_for_frame()`：可构造 `query2d_feats/query2d_pos`
- `QueryDecoder._apply_daca2d()`：SP 域 DACA mask + cross-attn 注入

必须修复/保证：
- `query2d_pos` 必须写入 decoder 的 **wo_elastic 距离空间**
- `sp_pos_list` 必须来自同一 wo_elastic 空间（不能用 elastic 后的点坐标）
- `in_box` support selection 必须使用 raw 投影（reverse aug）与 pose/intr 一致

### V3 实施方案（接口级）

1) 在 `extract_feat()` 返回值/中间缓存里，显式提供两套 SP 位置：
   - `sp_pos_wo_elastic`：用于 DACA‑2D 的距离门控（必需）
   - （可选）`sp_pos_elastic`：仅用于特征学习，不参与距离阈值
2) `_run_gdino_daca2d_for_frame()` 改为“两段式空间处理”：
   - raw 投影空间：用 `xyz_raw_for_proj` 选出 in_box 点索引
   - decoder 空间：用同一索引在 `xyz_aug_wo_elastic` 上取 median 得 `query2d_pos`
3) DACA‑2D 注入域固定：
   - `inject_domain="sp"`（你要求 SP 域注入）
   - P 域不注入（避免维度/口径漂移）
4) decoder 层顺序：保留两种 order 做消融：
   - paper：cross-attn → DACA‑2D → self-attn → FFN
   - code：cross-attn → self-attn → DACA‑2D → FFN

### V3‑2.1 关键对齐：mask.metric 必须与 SegDINO3D 一致（L1）
SegDINO3D 的 DACA‑2D 距离门控是：
- `dist = torch.cdist(pos_wo_elastic, dinox_query_pos, p=1)`（L1）
- `mask = dist <= 0.2`

因此 AutoSeg3D 的 `gdino_daca2d.mask` 必须对齐为：
- `mask.metric='l1'`
- `mask.thr=0.2`

并在日志/monitor 中固定输出：
- `nq_keep_mean / nq_pos_mean / qpos_rate_mean`（确保 object-level 的 query2d_pos 真在产生）
- `empty_mask_rows_rate`（确保 DACA mask 没把 attention 全置空）

## V3‑3：Decoder 与 bbox/positional embedding（对齐 SegDINO3D 的 box modulation/normalize）

SegDINO3D 的有效工程增强不止 DACA‑2D，还包含：
- `add_positional_embedding=True`（3D query/point positional embedding）
- `add_box_size_pred=True`（预测 size）
- `box_modulate_ca=True`（用 box size modulation cross-attn）
- `normalize_box_prediction=True`

### V3 实施策略（可落地、避免大改 SP/P 混合）

优先级建议：
1) **先把 center/size 的输出接口补齐**（`pred_centers/pred_sizes`）并打通 loss（见下一节）
2) 再实现 `box_modulate_ca/normalize_box_prediction`（需要在 cross-attn 层对 query_pos 做 modulation，属于结构改动，但逻辑可以直接参考 SegDINO3D）

注意：AutoSeg3D 的 decoder（`ScanNetMixQueryDecoder`）目前已有：
- `bbox_flag` 与 `out_reg (d_model→6)`（axis-aligned bbox）
但缺少：
- `add_positional_embedding/add_box_size_pred/box_modulate_ca/normalize_box_prediction` 这一整套

因此 V3 建议采用“显式对齐 SegDINO3D”的方式：
- 新增一个 decoder 分支/新 decoder 类，代码结构尽量贴近 `SegDINO3D/segdino3d/models/decoder/instance_seg_3d_decoder.py`
- 仍可复用 AutoSeg3D 现有的 SP/P 掩码预测（如果你决定保留 P 域 refine），但 **DACA‑2D 只作用于 SP 域层**

### V3‑3.1 最小可落地的 decoder 深度对齐（先做 ablation）
在不立刻重写 decoder 的前提下，先做一组“最小深度对齐”验证：
- `ScanNetMixQueryDecoder.num_layers: 3 → 6`
- 同时确保 `mask_pred_mode` 至少在前若干层为 `SP`（否则 SP 域不会触发 DACA‑2D 注入）
- 保持原 3D cross‑attn/self‑attn 逻辑不变，DACA‑2D 只作为额外模块插入（不会改变主干注意力）

这一步的目的不是“直接达到最佳”，而是快速回答一个关键问题：  
**2D 注入在 3 层 decoder 上会不会天然用不起来？如果 6 层明显更好，再投入实现 box‑modulated CA 等更重改动。**

## V3‑4：Loss/Matcher（对齐 SegDINO3D：CenterL1Cost + SizeL1Cost + loss_weight）

### SegDINO3D 的事实（必须对齐）

在 `SegDINO3D/configs/prototypes/SegDINO3D_ScanNet200.py`：
- matcher costs 增加：
  - `CenterL1Cost(weight=0.5)`
  - `SizeL1Cost(weight=0.5)`
- `loss_weight = [0.5, 1.0, 1.0, 0.5, 0.5, 0.5]`
- `add_positional_embedding=True`
- `decoder_cfg.add_box_size_pred=True`
- `decoder_cfg.box_modulate_ca=True`
- `decoder_cfg.normalize_box_prediction=True`

### GT center/size 的来源（不是数据集字段）

SegDINO3D 在训练时由 GT mask + 点坐标计算：
- `instance_centers (n_inst,3)`
- `instance_sizes (n_inst,3)`
对应实现：`SegDINO3D/segdino3d/models/architecture/baseline3d.py:get_extra_instance_data()`

### V3 实施方案（接口级）

1) 在 AutoSeg3D 的训练 forward（loss 前）增加等价的 `get_extra_instance_data()`：
   - 输入：GT instance masks（按你 SV 的 GT 表示） + 点坐标（必须用 `pos_wo_elastic`）
   - 输出：写入 `data_sample.gt_instances.instance_centers/instance_sizes`
2) matcher 增加 `CenterL1Cost/SizeL1Cost`；loss 增加对应项
3) `loss_weight` 按 SegDINO3D 对齐（并明确每一项对应哪个 loss，避免长度对不上）

### V3‑4.1 center/size GT 的“运行时计算”定义（对齐 SegDINO3D）
SegDINO3D 不是依赖数据集 bbox，而是在训练时由 GT mask + 点坐标计算：
- 对每个 GT instance，取其点集 `P = {x_k ∈ R^3}`（使用与 decoder 一致的 `pos_wo_elastic` 空间）
- `center = (P.min(dim=0) + P.max(dim=0)) / 2`（axis-aligned bbox center）
- `size = P.max(dim=0) - P.min(dim=0)`（axis-aligned bbox size）

写入到 targets 后：
- matcher cost 用 `L1(center_pred, center_gt)` 与 `L1(size_pred, size_gt)` 参与 assignment
- loss 用对应的 L1 监督（权重参考 SegDINO3D 的 loss_weight 第 5/6 项）

> 这一步的意义：让 3D queries 学到更稳定的几何对齐能力（尤其在你只做 SP 域 DACA 注入时，几何对齐更关键）。

## V3‑5：你要的新 SV 配置文件（建议命名与关键参数）

建议新增（不要污染旧 config）：
- `AutoSeg3D/configs/scannet200/AutoSeg3D_sv_scannet200_segdino3d_v3.py`

关键差异参数（相对 routeA_gdino）：
- `gdino_point_fusion.out_dim = 256`（不再压缩）
- `backbone.in_channels = 259`
- `gdino_daca2d.enable=True` 且 `mode="fuse"`（训练也开启）
- decoder：启用 positional embedding + center/size 预测 + box modulation（按 SegDINO3D 对齐）
- criterion/matcher：加入 Center/Size 监督与 loss_weight 对齐

## V3‑6：最小自测（训练前必须通过）

在正式开始训练前，必须一次性通过下面的 sanity（只跑 1–2 iter 即可）：
- `gdino_point_fusion.valid_ratio_mean >= 0.99`（每帧）
- early fusion 维度一致：`rgb(3)+2d(256)=259`，UNet 第一层不报维度 mismatch
- `query2d_pos` 的 `query_pos_valid_rate` 不为 0（否则 DACA 退化）
- DACA mask 不全空：`empty_mask_rows_rate` 低且 dummy query 生效
- loss 端：center/size loss 不为 NaN，且 matched 数量合理
