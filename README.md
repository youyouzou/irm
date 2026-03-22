# WiFi CSI 上的 ERM / IRM 域泛化（IRM DG v2）

本项目用于在 WiFi CSI 分类任务上对比 `ERM` 与 `IRM`。  
当前版本重点是：**同条件配对实验**，确保可比性与科研记录一致性。

---

## 1. 核心改进（已实现）

### 1.1 统一实验记录（可复现）
每次训练都会在 `output_dir` 自动生成：

- `resolved_config.json`：训练开始时的完整生效配置
- `logs.csv`：按 epoch 记录训练日志
- `run_summary.json`：训练结束后的最终摘要（含 test）

并自动追加到全局索引：

- `experiments/experiment_index.csv`

### 1.2 统一摘要字段（run_summary.json）
固定字段包含：

- `algorithm`
- `model_variant`
- `seed`
- `best_epoch`
- `selected_val_acc`
- `val_acc`
- `val_worst_env_acc`
- `test_acc_selected`
- `test_acc_overall`
- `test_acc_worst_env`
- `checkpoint_paths`
- `timestamp`

### 1.3 CNN 结构改进（可开关，兼容旧模型）
`models/cnn.py` 支持：

- `model_variant=baseline`（默认）
- `model_variant=msstem`

`msstem` 为多尺度时间卷积 stem：

- 时间核并行：`k_t = {3, 7, 11}`
- 频域核固定：`3`
- concat 后 `1x1` 融合

### 1.4 同条件配对对比（ERM vs IRM）
新脚本 `scripts/run_irm_dgv2_plan.py` 默认会对每个 IRM run 自动配一个 ERM run（同 seed、同增强、同 lr/wd/epochs/model_variant），并输出配对对比结果。

---

## 2. 目录（核心）

```text
IRM/
├─ data/
│  └─ processed/
├─ models/
│  └─ cnn.py
├─ train/
│  ├─ trainer.py
│  ├─ train_erm.py
│  ├─ train_irm.py
│  ├─ config.erm_dg.json
│  ├─ config.irm_dg.json
│  └─ config.example.json
├─ scripts/
│  ├─ eval_checkpoint.py
│  ├─ plot_results.py
│  └─ run_irm_dgv2_plan.py
├─ docs/
│  └─ IRM_DG_v2.md
└─ experiments/
   └─ experiment_index.csv
```

---

## 3. 数据格式要求

`.npz` 至少包含：

- `x`: `[N, H, W]` 或 `[N, 1, H, W]`（`float32`）
- `y`: `[N]`（`int64`）
- `env`: `[N]`（`int64`）

默认路径：

- `data/processed/train.npz`
- `data/processed/val.npz`
- `data/processed/test.npz`

---

## 4. 环境安装

```bash
pip install -r requirements.txt
```

若使用 YAML 配置：

```bash
pip install pyyaml
```

---

## 5. 关键参数与实验设置

### 5.1 常用参数
- `num_classes`
- `batch_size`
- `epochs`
- `lr`
- `weight_decay`
- `dropout`
- `model_variant`（`baseline` / `msstem`）
- `seed`

### 5.2 IRM 参数
- `irm_lambda`
- `penalty_anneal_epochs`
- `penalty_ramp_epochs`
- `val_selection_metric`（`overall` / `worst_env` / `hybrid`）

### 5.3 数据增强参数
- `input_norm`
- `use_augment`
- `aug_prob`
- `amp_scale_low`, `amp_scale_high`
- `noise_std_ratio`
- `time_mask_ratio`
- `subcarrier_mask_ratio`

### 5.4 v2 计划固定公共设置
- `batch_size=16`
- `epochs=140`
- `dropout=0.45`
- `label_smoothing=0.0`
- `val_selection_metric=hybrid`（IRM）
- `aug_prob=0.55`
- `amp_scale=[0.85, 1.15]`
- `noise_std_ratio=0.01`
- `time_mask_ratio=0.05`
- `subcarrier_mask_ratio=0.08`

### 5.5 Stage A 参数网格（IRM 核心）

| 组别 | lr | weight_decay | irm_lambda | anneal | ramp |
|---|---:|---:|---:|---:|---:|
| A1 | 3.5e-4 | 1e-4 | 1.5 | 10 | 20 |
| A2 | 3.5e-4 | 1e-4 | 2.0 | 10 | 20 |
| A3 | 3.0e-4 | 1e-4 | 2.5 | 10 | 20 |
| A4 | 3.0e-4 | 1e-4 | 3.0 | 10 | 20 |
| A5 | 3.5e-4 | 1e-4 | 2.0 | 5 | 15 |
| A6 | 3.0e-4 | 1e-4 | 2.5 | 5 | 15 |
| A7 | 3.5e-4 | 3e-4 | 2.0 | 10 | 30 |
| A8 | 3.0e-4 | 3e-4 | 2.5 | 10 | 30 |

Stage B：
- 取 Stage A 中 IRM `selected_val_acc` 前 2（并列用 `val_worst_env_acc` 破）
- 切换 `model_variant=msstem`
- 同条件再跑配对 ERM

Stage C：
- 取 Stage B 最优 IRM 配置
- 跑 `seed=7` 与 `seed=99`
- 同条件继续配对 ERM

---

## 6. 运行命令

### 6.1 单次训练（配置驱动）

ERM：

```bash
python -m train.train_erm --config train/config.erm_dg.json
```

IRM：

```bash
python -m train.train_irm --config train/config.irm_dg.json
```

> `num_classes` 可以写在配置文件中；命令行参数优先级更高。

### 6.2 覆盖参数训练示例

```bash
python -m train.train_irm \
  --config train/config.irm_dg.json \
  --model_variant msstem \
  --irm_lambda 2.5 \
  --penalty_anneal_epochs 10 \
  --penalty_ramp_epochs 20 \
  --val_selection_metric hybrid \
  --output_dir experiments/my_irm_run
```

### 6.3 评估 checkpoint（自动识别 baseline/msstem）

```bash
python scripts/eval_checkpoint.py --checkpoint experiments_gpu5/irm_dg_tuned/best_model.pt
```

### 6.4 运行配对实验计划（推荐）

先 dry-run 看将执行哪些命令：

```bash
python -m scripts.run_irm_dgv2_plan --dry_run
```

正式执行（默认成对跑 ERM+IRM）：

```bash
python -m scripts.run_irm_dgv2_plan \
  --irm_config train/config.irm_dg.json \
  --erm_config train/config.erm_dg.json \
  --output_root experiments/irm_dgv2 \
  --device cuda
```

> 说明：该计划会跑 `12 组参数 × 2 算法`，总计约 `24` 次训练。

如仅跑 IRM（不建议，仅排障用）：

```bash
python -m scripts.run_irm_dgv2_plan --skip_erm
```

---

## 7. 配对实验输出说明

`--output_root` 下会生成：

- `plan_results.json`：所有 run 的完整记录（含 overrides + summary）
- `plan_results.csv`：所有 run 的扁平表
- `paired_compare.csv`：按 pair（同 stage/同参数）对比 IRM vs ERM
- `final_report.json`：最终统计报告

`paired_compare.csv` 关键列：

- `irm_test_acc_selected`
- `erm_test_acc_selected`
- `test_acc_gap_irm_minus_erm`

`final_report.json` 包含：

- `paired_summary`（IRM 胜出次数 / ERM 胜出次数 / 平均差值 / 标准差）
- `irm_group_metrics`（Stage B 最优 + Stage C 两个 seed）
- `erm_group_metrics`（对应 ERM 配对组）

---

## 8. 可视化与摘要

```bash
python scripts/plot_results.py \
  --erm_logs experiments/erm/logs.csv \
  --irm_logs experiments/irm/logs.csv \
  --output_dir experiments/plots
```

若存在 `run_summary.json`，`summary.txt` 会附加最终 test 指标摘要。

---

## 9. 推荐科研流程（强调可比性）

1. 固定数据划分和种子。  
2. 每次改动都成对跑 ERM/IRM。  
3. 用 `paired_compare.csv` 判断“IRM 是否稳定优于 ERM”。  
4. 用 `final_report.json` 报告 mean/std 与胜率。  
5. 用 `experiment_index.csv` 做长期全局追踪。

---

## 10. 常见问题

- Q: 为什么不记录每个 epoch 的 test？  
  A: 为避免测试集信息泄漏，只在训练结束后记录 test。

- Q: 旧 checkpoint 还能评估吗？  
  A: 可以。缺少 `model_variant` 时默认按 `baseline` 处理。

- Q: 没有 GPU 可以跑吗？  
  A: 可以，命令中加 `--device cpu` 即可（速度较慢）。
