# TAAC 2026 比赛方案

本仓库整理了一条 TAAC 2026 比赛中的可追溯改进方案主线，内容包括训练代码、评测代码、公开榜结果、实验演化记录、负结果整理和参考资料。

本仓库发布的是相对官方 baseline 的改进版本，不是官方 baseline 本身，也不保证在任意硬件、数据切分或运行环境下复现完全相同的公开榜分数。文档中的结论均限定于本次比赛数据、当时的验证方式、训练环境和公开榜记录。

## 适用范围

本仓库适合用于：

- 了解 TAAC 2026 转化率预估任务中的一种工程方案组织方式；
- 复查样本级元信息、离散日历时间和训练策略对公开榜结果的影响；
- 参考 train/eval 配对、实验记录和负结果归档方式；
- 在具备比赛数据和相近运行环境的前提下尝试复现。

本仓库不适合用于：

- 作为官方 baseline 或标准答案；
- 直接推断任意数据集上的推荐系统最佳实践；
- 在缺少比赛数据、schema 和 checkpoint 的情况下开箱即用；
- 作为不依赖比赛数据和运行环境的通用推荐系统框架。

## 方案概览

官方 baseline 是本仓库的对照起点。公开方案保留同一任务口径和 train/eval 配对思路，主要在以下四类位置做改进：

1. **保守的非序列 token 组织**：用户/物品离散特征被压缩为固定数量的 NS token，用户侧 dense 字段按结构处理后仍融合为单个 dense token，避免随意增加 token 数。
2. **样本级摘要特征工程**：补充序列长度、截断强度、recency、span、padding/sentinel 密度、列表字段覆盖情况等样本级摘要信息，缓解固定长度序列和缺失值处理造成的信息压缩。
3. **样本级元信息弱条件化**：在特征工程父线基础上，将上述摘要信息和离散日历时间通过 query 生成前的上下文和最终表示残差两条路径进入模型。
4. **稳定主干上的训练策略强化**：保持 `v24-6` 的特征与模型路径不变，合入 `bce_pairwise`、cosine 学习率调度、warmup、EMA、label smoothing 和 dense weight decay 等训练策略，形成最终公开版本。

其中，`v12`、`v24`、`v24-6`、`v87` 是历史实验记录中的版本号：

| 版本 | 可公开理解的含义 |
| --- | --- |
| `v12` | 样本级摘要特征工程父线 |
| `v24` | 早期样本级元信息条件化版本 |
| `v24-6` | 在 `v24` 上加入样本级离散日历时间特征的版本 |
| `v87` | 保持 `v24-6` 特征和模型主线不变，强化训练策略后的版本 |

本开源包默认对应 `v87` 这一训练策略强化版本，也是当前整理材料中公开 AUC 最高的主线版本。

## 相对官方 baseline 的主要提升

| 改进阶段 | 公开 AUC | 相对官方 baseline | 说明 |
| --- | ---: | ---: | --- |
| 官方 baseline | `约 0.812` | `+0.0000` | 官方默认方案 |
| 样本级摘要特征工程版本 | `0.816` | `约 +0.0040` | 补充长度、截断、recency、span 和缺失模式等 meta-summary 特征 |
| 早期样本级元信息条件化版本 | `0.821977` | `约 +0.0100` | 样本级元信息提前参与 query 生成 |
| 离散日历时间版本 | `0.825905` | `约 +0.0139` | 在元信息通道中加入 hour、weekday、hour-weekday 特征 |
| 训练策略强化版本 | `0.828537` | `约 +0.0165` | 在不替换特征与模型主线的前提下强化训练策略 |

## 最高分版本相对官方 baseline 的修改

当前开源包对应的最高分版本为 `v87`，公开 AUC 为 `0.828537`。它不是对官方 baseline 的简单参数调整，而是在数据表示、模型信息路径和训练策略上做了连续改造。

| 模块 | 相对官方 baseline 的主要修改 | 目的 |
| --- | --- | --- |
| 数据与特征 | 构造 `meta_feats`，加入序列长度、截断强度、recency、span、padding/sentinel 密度、列表字段覆盖情况和粗粒度日历时间 | 补充固定长度序列和缺失值处理过程中容易被压平的样本级状态信息 |
| 用户 dense 表示 | 将用户 dense 字段按结构拆分处理，再融合为单个用户 dense token；重点处理单字段和与用户离散字段对齐的 dense 组 | 保留 dense 字段内部结构，同时不额外扩大主干 token 数 |
| 非序列离散特征 | 使用固定数量 NS token 汇总用户/物品离散特征，公开配置为用户侧 `5` 个、物品侧 `2` 个 | 控制 token 几何，减少随意增加 token 对主干优化的扰动 |
| 序列建模 | 多行为域分别编码，使用时间 bucket embedding，并由每个行为域生成 query token 做 cross-attention | 在统一框架内聚合不同域的行为序列信息 |
| 样本级元信息路径 | `meta_feats` 同时进入早期 query 生成和最终表示残差，而不是只在分类头附近使用 | 让样本级上下文参与“如何读取序列”的过程 |
| 粗粒度时间上下文 | 将 hour、weekday、hour-weekday one-hot 作为样本级元信息输入，而不是建立单独的强时间分支 | 使用稳定的时间上下文，避免 direct time-logit 等更强路径带来的不稳定 |
| 训练策略 | 启用 `bce_pairwise`、`pairwise_lambda=0.05`、cosine 学习率调度、`warmup_steps=500`、`EMA(decay=0.999)`、`label_smoothing=0.01`、dense `weight_decay=0.02` | 在稳定特征和模型主线上改善排序目标、训练平滑性和正则化 |
| 评测部署 | 训练和评测代码配对整理，评测侧默认使用 PyTorch SDPA 路径，并从 checkpoint 的 `train_config.json` 还原关键结构配置 | 降低评测环境依赖，避免 train/eval 配置不一致 |

从实验演化看，`v24-6` 已经提供有效的特征与模型主线，公开 AUC 为 `0.825905`；`v87` 在保持 `v24-6` 的数据特征、模型结构、token 几何、早期元信息条件化和最终元信息残差不变的前提下，主要合入训练策略强化，将公开 AUC 提升到 `0.828537`。因此，最高分版本的新增收益更应理解为“强特征主线上的训练策略提升”，而不是重新设计了一套更复杂的输入特征。

## 公开结果

下表列出的是历史记录中能够较清楚对应到公开榜的阶段性结果，其中 hash fallback 是输入侧容量对照，不是最终选择的开源主线。

| 阶段 | 公开 AUC | 主要变化 |
| --- | ---: | --- |
| 官方 baseline | `约 0.812` | 官方默认方案 |
| 样本级摘要特征工程版本 | `0.816` | 补充序列长度、截断、recency、span 和缺失模式等 meta-summary 特征 |
| 早期样本级元信息条件化版本 | `0.821977` | 将样本级元信息提前用于 query 生成 |
| 离散日历时间版本 | `0.825905` | 在元信息通道中加入 hour、weekday、hour-weekday 特征 |
| 训练策略强化版本 | `0.828537` | 保持模型和特征主线不变，强化损失、调度与稳定性策略 |
| hash fallback 对照版本 | `0.827938` | 恢复部分被跳过的高基数离散特征，用于对照输入侧容量 |

最终开源选择“训练策略强化版本”，原因是该版本同时具备较好的公开榜结果、清楚的训练/评测配对关系，以及较完整的实验解释。

排名轨迹和分数背景见 [docs/results_and_rank.md](./docs/results_and_rank.md)。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [train/](./train) | 训练入口、模型、数据集、trainer 和启动脚本 |
| [eval/](./eval) | 评测入口、评测数据集和与训练配置对应的推理模型 |
| [assets/](./assets) | 文档使用的公开榜排名轨迹文件 |
| [docs/model_overview.md](./docs/model_overview.md) | 模型结构与关键设计说明 |
| [docs/experiment_summary.md](./docs/experiment_summary.md) | 主线实验演化总结 |
| [docs/experiment_catalog.md](./docs/experiment_catalog.md) | 更完整的实验方向目录 |
| [docs/negative_results.md](./docs/negative_results.md) | 代表性负结果和边界说明 |
| [docs/results_and_rank.md](./docs/results_and_rank.md) | 公开分数与排名背景 |
| [docs/release_notes.md](./docs/release_notes.md) | 开源范围、复现边界和发布检查项 |
| [docs/terminology.md](./docs/terminology.md) | 文档术语说明 |
| [docs/references.md](./docs/references.md) | 论文和外部仓库参考 |

## 快速开始

### 运行假设

代码沿用比赛环境中的若干约定：

- 数据以 Parquet 文件组织，并包含对应的 `schema.json`。
- 训练和评测分别在 `train/` 与 `eval/` 目录中执行。
- 数据、checkpoint、日志和输出路径通过环境变量传入。
- 训练默认优先使用可用的 FlashAttention 路径；评测默认强制使用 PyTorch SDPA 路径，以减少部署依赖。

主要 Python 依赖包括 PyTorch、NumPy、PyArrow、tqdm 和 scikit-learn。

### 训练

```bash
cd train
export TRAIN_DATA_PATH=/path/to/train_data
export TRAIN_CKPT_PATH=/path/to/checkpoints
export TRAIN_LOG_PATH=/path/to/logs
bash run.sh
```

训练脚本中的关键环境变量包括：

- `TAAC_VALID_RATIO`
- `TAAC_LOSS_TYPE`
- `TAAC_PAIRWISE_LAMBDA`
- `TAAC_LR_SCHEDULE`
- `TAAC_WARMUP_STEPS`
- `TAAC_EMA_DECAY`
- `TAAC_LABEL_SMOOTHING`
- `TAAC_WEIGHT_DECAY`
- `TAAC_ATTN_BACKEND`

完整默认值以 [train/run.sh](./train/run.sh) 为准。

### 评测

```bash
cd eval
export MODEL_OUTPUT_PATH=/path/to/checkpoint/global_step_x
export EVAL_DATA_PATH=/path/to/eval_data
export EVAL_RESULT_PATH=/path/to/result_dir
python infer.py
```

`MODEL_OUTPUT_PATH` 应指向包含 `model.pt` 与 `train_config.json` 的 checkpoint 目录。评测结果会写入 `EVAL_RESULT_PATH`。

## 包含内容

本仓库包含：

- 一组配对的训练与评测代码；
- 与公开主线直接相关的模型说明；
- 可追溯的阶段性公开榜结果；
- 主线实验演化记录；
- 代表性负结果；
- 比赛期间参考过的论文与外部仓库信息。

## 不包含内容

本仓库不包含：

- 全量历史实验分支；
- 已训练 checkpoint；
- 比赛平台账号、缓存和中间产物；
- 仅用于短期验证或提交试探的临时脚本；
- 结果来源无法清楚确认的私有尝试。

## 推荐阅读顺序

如果希望理解方案结构，建议按以下顺序阅读：

1. [docs/model_overview.md](./docs/model_overview.md)
2. [docs/terminology.md](./docs/terminology.md)
3. [docs/experiment_summary.md](./docs/experiment_summary.md)
4. [docs/experiment_catalog.md](./docs/experiment_catalog.md)
5. [docs/negative_results.md](./docs/negative_results.md)

如果希望尝试复现，建议先阅读：

1. [train/run.sh](./train/run.sh)
2. [train/train.py](./train/train.py)
3. [eval/infer.py](./eval/infer.py)
4. [docs/release_notes.md](./docs/release_notes.md)

## 反馈与贡献

如启用 GitHub Issues，建议将复现问题、文档问题和结果口径问题分开提交，并在问题描述中提供运行环境、数据路径组织方式、关键环境变量和完整报错日志。涉及比赛平台账号、私有路径或未公开数据的信息不应提交到公开 issue。
