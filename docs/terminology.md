# 术语说明

本文档统一说明仓库中使用的核心术语，避免模型说明、实验记录和复现说明之间出现表达偏差。

## 样本级元信息

样本级元信息指描述当前样本整体状态的摘要特征，而不是某条行为序列或某个静态离散字段本身。

典型例子包括：

- 序列长度和截断统计；
- recency 与时间跨度摘要；
- padding、sentinel 或缺失模式统计；
- 粗粒度日历时间特征。

历史实验记录中有时会用 `meta_feats` 指代这组特征。

在主线演化中，`v12` 首先验证了这类样本级摘要特征工程的收益，公开 AUC 为 `0.816`；后续 `v24` 在此基础上进一步调整元信息进入模型的位置。

## 早期样本级元信息条件化

早期样本级元信息条件化指：在 query 生成阶段使用样本级元信息，而不是只在最终分类头之前使用。

这一设计对应历史记录中的 `early meta-query` 或 `meta query` 路线。

## 用户侧 dense 表示 token

用户侧 dense 表示 token 指将用户侧 dense 特征按来源和结构整理后形成的单个 token。

它不是将所有 dense 字段直接拼接后投影，也不是把 dense 字段展开成多个额外 token。后续负结果显示，在本方案主线上，保持单个结构化 dense token 比扩展为多 token 更稳定。

## NS Token

NS token 指由非序列特征构造出的 token，包括用户侧离散特征 token、物品侧离散特征 token，以及用户/物品 dense token。

在公开配置中，用户侧离散特征被压缩为 `5` 个 NS token，物品侧离散特征被压缩为 `2` 个 NS token。用户 dense 和物品 dense 如存在，则各自作为额外非序列 token 加入。

## RankMixer

RankMixer 是模型中用于固定 token 表交互的模块。它接收行为域 query token 和 NS token 拼接后的 token 表，通过 token mixing 与逐 token FFN 更新这些 token。

在 `full` 模式下，RankMixer 要求总 token 数能够整除 `d_model`。因此，改变 query 数、NS token 数或 dense token 数会影响主干几何，不应被视为无关紧要的表面改动。

## Query 生成

Query 生成指从多域行为序列的 pooled summary 中生成 query token，并用这些 query token 进一步聚合各行为域信息的过程。

文档中提到“元信息提前影响 query 生成”时，均指这一位置。

## 元信息残差

元信息残差指将 `meta_feats` 投影后，以 gated residual 形式加到最终样本表示上的路径。

它与早期样本级元信息条件化不同：前者作用在输出表示附近，后者作用在 query 生成阶段。公开主线同时使用这两条弱条件化路径。

## 粗粒度日历时间

粗粒度日历时间指以下离散时间特征：

- 小时；
- 星期；
- 小时与星期的组合。

在本方案中，这些特征通过样本级元信息通道进入模型，而不是作为单独的强时间分支。

## 训练策略强化

训练策略强化指保持模型主干和特征工程基本不变，仅调整训练侧策略。最终公开版本中的主要训练策略包括：

- `bce_pairwise`；
- pairwise loss 权重；
- cosine 学习率调度；
- warmup；
- EMA；
- label smoothing；
- dense 参数 weight decay。

这一路线对应历史记录中的 `v87`。
