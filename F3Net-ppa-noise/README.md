# PPA Loss 在标注噪声下的脆弱性研究

> 从 F3Net 大作业主线中独立出来的一个研究方向（2026-06，原 commit `e3fefa0`）。
> 主线（F3Net 复现 + ssim/边界锐化加分项）仍在 `../F3Net/`。

## 核心假说

F3Net 的 PPA 损失对每个像素加权 `w = 1 + γ·α`，`α = |avgpool(GT,31) − GT|`，在**边界**处最大。
而边界恰是人工标注最不确定的地方 → **"越不确定越强调"的恶性循环**：标注噪声被 PPA 的边界加权放大，
模型过拟合错误的边界标签。机制类比 focal/OHEM 对 noisy-label 的脆弱性。

## 零训练证据（`analyze_ppa_weight.py`）

边界带占 **6.0%** 像素，却吃掉 **14.5%** 的 PPA 权重（2.42× 过度集中）；单像素放大 **2.74×**，
小目标更狠 **2.99×**。即 PPA 把约 2.4–3 倍权重压在"将来会被噪声污染"的边界像素上。

## 主结果：噪声强度扫描（噪声只注入训练 GT，干净测试集评测）

| 噪声率 | PPA MAE | BCE MAE | PPA优势(MAE) | PPA F_mean | BCE F_mean | PPA优势(F_mean) | PPA BMAE | BCE BMAE |
|--------|---------|---------|------|-----------|-----------|------|------|------|
| 0% (干净) | 0.0435 | 0.0496 | **+0.0061** | 0.9010 | 0.8898 | **+0.0112** | 0.1751 | 0.2121 |
| 2.04% | 0.0521 | 0.0564 | +0.0043 | 0.8740 | 0.8702 | +0.0038 | 0.2604 | 0.3150 |
| 3.0% | 0.0610 | 0.0624 | +0.0014 | 0.8604 | 0.8580 | +0.0024 | 0.3651 | 0.3760 |
| **4.2%** | 0.0678 | 0.0672 | **−0.0006** | 0.8289 | 0.8402 | **−0.0113** | **0.5043** | 0.4480 |

- **PPA 优势单调崩塌并穿越零线**：噪声越大，PPA 相对 BCE 的优势越小，4.2% 噪声处 **PPA 反输 BCE**。
- **🎯 边界自毁（最强证据）**：4.2% 时 PPA 的 Boundary MAE（0.5043）反而**比不管边界的 BCE（0.4480）更差**；
  PPA BMAE 从干净的 0.1751 爆炸到 0.5043（~3×）——强调边界的损失把自己最该擅长的边界给毁了。

## 复现

```bash
# 1. 零训练证据
python analyze_ppa_weight.py --datapath ./data --split train
# 2. 生成噪声训练 GT（只动 train，test 保持干净）
python make_noisy_gt.py --datapath ./data --width 5 --prob 0.7 --seed 42   # -> data/gt_noisy_d5p70/
# 3. 在噪声 GT 上训练（--mask_subdir 指向噪声目录），干净测试集评测
python train.py --train_split train --mask_subdir gt_noisy_d5p70 --loss ppa --workers 0 --savepath ./checkpoints_noise/ppa_d5p70
python evaluate.py --checkpoint ./checkpoints_noise/ppa_d5p70/model_epoch200.pth --workers 0
# 4. 可视化干净 vs 噪声 GT
python viz_noisy_gt.py --noisy_dir gt_noisy_d3p50 --out noise_viz/compare.png
```

已训练 checkpoints 在 `checkpoints_noise/{ppa,bce}_{d3p50,d5p50,d5p70}/`，评测日志在 `improve_logs/noise_*.txt`。

## 待办（升级成小论文的拼图）

- **γ 消融**（关键，证 PPA 特有）：固定 d5p70，γ={3,5,8}，看 γ 越大退化越狠 → 坐实是加权机制在放大噪声。
- **修复方案**：γ 退火 / 不确定性感知地 down-weight 不一致的边界像素（反 PPA 而行）。
- **真实锚点**：PASCAL-S 多标注者，算 α 与标注者分歧的相关性。
- **子集分析**：细长/小目标子集上崩塌最剧烈（建议换 DUT-OMRON，ECSSD 小目标样本太少）。
- 严格性：matched clean 基线用相同 code（workers=0）重训，消除与 noisy 点的微小训练差异。
