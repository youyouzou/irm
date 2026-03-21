# IRM / ERM for WiFi CSI Domain Generalization

本项目用于在 WiFi CSI 数据上对比 `ERM` 与 `IRM`，并提供统一训练框架、配置文件驱动和实验结果分析工具。

## 1. 当前能力概览

- 统一训练核心：`train/trainer.py`
  - `run_erm(args)` 和 `run_irm(args)` 共用评估、日志、早停、学习率调度等逻辑
- 训练入口（轻量包装）
  - `train/train_erm.py`
  - `train/train_irm.py`
- 配置文件支持（JSON / YAML）
  - `--config` 参数先加载配置，再由命令行参数覆盖
  - 工具：`train/config_utils.py`
  - 示例：`train/config.example.json`
- 模型
  - `models/cnn.py`：改进版 CSIClassifier（残差块 + GroupNorm + SE）
- 可视化分析
  - `scripts/plot_results.py`：生成曲线图 + `summary.txt` 摘要

## 2. 目录结构（核心）

```text
IRM/
├─ data/
│  └─ processed/                  # train.npz / val.npz / test.npz
├─ datasets/
│  └─ wifi_dataset.py
├─ losses/
│  └─ irm_loss.py
├─ models/
│  └─ cnn.py
├─ train/
│  ├─ trainer.py                  # 统一训练器
│  ├─ train_erm.py                # ERM 入口
│  ├─ train_irm.py                # IRM 入口
│  ├─ config_utils.py             # 配置加载
│  └─ config.example.json         # 配置示例
├─ scripts/
│  ├─ make_npz_from_npy.py
│  └─ plot_results.py
└─ experiments/
   ├─ erm/
   ├─ irm/
   └─ plots/
```

## 3. 数据格式要求

每个 `.npz` 文件至少包含：

- `x`：`[N, H, W]` 或 `[N, 1, H, W]`，`float32`
- `y`：`[N]`，类别标签，`int64`
- `env`：`[N]`，环境标签，`int64`

默认路径：

- `data/processed/train.npz`
- `data/processed/val.npz`
- `data/processed/test.npz`

> IRM 训练要求 `train.npz` 中有多个环境（`env` 至少两个不同值）。

## 4. 环境安装

```bash
pip install -r requirements.txt
```

如果使用 YAML 配置文件，需要额外安装：

```bash
pip install pyyaml
```

## 5. 快速开始

### 5.1 使用配置文件训练（推荐）

先复制并修改示例配置：`train/config.example.json`

ERM：

```bash
python -m train.train_erm --config train/config.example.json --num_classes 7
```

IRM：

```bash
python -m train.train_irm --config train/config.example.json --num_classes 7
```

用eval_checkpoint.py加载保存的模型跑测试集的命令：
```bash
python -u scripts/eval_checkpoint.py --checkpoint experiments_gpu5/irm_dg_tuned/best_model.pt
```


### 5.2 直接命令行训练

ERM：

```bash
python -m train.train_erm \
  --num_classes 7 \
  --train_npz data/processed/train.npz \
  --val_npz data/processed/val.npz \
  --test_npz data/processed/test.npz \
  --epochs 80 \
  --device cpu \
  --output_dir experiments/erm
```

IRM：

```bash
python -m train.train_irm \
  --num_classes 7 \
  --train_npz data/processed/train.npz \
  --val_npz data/processed/val.npz \
  --test_npz data/processed/test.npz \
  --irm_lambda 10 \
  --penalty_anneal_epochs 10 \
  --epochs 80 \
  --device cpu \
  --output_dir experiments/irm
```

> 命令行参数优先级高于配置文件。

## 6. 输出内容

每次训练会在 `output_dir` 下生成：

- `logs.csv`：训练日志
- `best_model.pt`：验证集最优模型

日志字段：

- ERM：`epoch, train_loss, train_acc, val_acc, ema_val_acc, selected_val_acc, lr`
- IRM：`epoch, total_loss, erm_loss, penalty, val_acc, irm_weight, lr`

## 7. 结果可视化与分析

```bash
python scripts/plot_results.py \
  --erm_logs experiments/erm/logs.csv \
  --irm_logs experiments/irm/logs.csv \
  --output_dir experiments/plots \
  --smooth_window 5
```

输出文件（典型）：

- `erm_accuracy.png`
- `erm_loss_lr.png`
- `irm_accuracy.png`
- `irm_loss_penalty.png`
- `erm_vs_irm_compare.png`
- `summary.txt`

`summary.txt` 会给出：

- best / last 精度
- tail mean（末段均值）
- tail std（末段波动）
- ERM 与 IRM 的差值对比

## 8. 常用调参建议

- ERM 先稳住基线：
  - `lr`, `weight_decay`, `ema_decay`, `label_smoothing`, `early_stop_patience`
- IRM 重点调：
  - `irm_lambda`, `penalty_anneal_epochs`, `lr`, `batch_size`
- 若验证集波动大：
  - 降低 `lr`
  - 增大 `batch_size`（资源允许）
  - 增大 `early_stop_patience`

## 9. 一次完整实验建议流程

1. 准备/检查 `train.npz, val.npz, test.npz`
2. 运行 ERM，得到基线
3. 运行 IRM，保持同数据划分
4. 运行 `plot_results.py` 生成图和摘要
5. 根据 `summary.txt` 决定下一轮调参方向

## 10. 备注

- 若出现 `Torch not compiled with CUDA enabled`，请改用 `--device cpu` 或安装 CUDA 版本 PyTorch。
- 如果你希望，我可以继续补一个“批量实验汇总脚本”，自动对比多个 `experiments/*/logs.csv`。
