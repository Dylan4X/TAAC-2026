# 模型概览

## 定位

公开版本使用的模型名为 `PCVRHyFormer`，面向 TAAC 2026 中的转化率预估任务。它是相对官方 baseline 的改进版本，而不是官方 baseline 的再发布。

与官方 baseline 相比，本版本的改动重点不在改变任务定义或输出格式，而在输入表示组织、样本级上下文使用方式和训练策略上。模型输入由五类信息组成：

- 用户侧离散特征；
- 物品侧离散特征；
- 用户侧 dense 特征；
- 物品侧 dense 特征；
- 多个行为域的序列特征及其时间 bucket。

模型的目标不是提出一个通用推荐框架，而是在官方 baseline 所定义的比赛任务上，把静态字段、dense 字段、行为序列和样本级上下文放到更稳定的表示路径中。

## 相对官方 baseline 的改进点

相对官方 baseline，本版本主要增加或强化了以下能力：

| 改进点 | 作用 |
| --- | --- |
| 结构化用户 dense token | 按字段结构处理用户侧 dense 信息，再融合回单个 dense token |
| 固定数量 NS token | 将用户/物品离散特征压缩为稳定 token 表，控制主干 token 几何 |
| 样本级摘要特征工程 | 补充长度、截断、recency、span 和缺失模式等 meta-summary 特征 |
| 样本级元信息路径 | 将样本级摘要信息作为上下文使用 |
| 早期 query 条件化 | 让样本级元信息在 query 生成阶段参与序列读取 |
| 离散日历时间特征 | 将 hour、weekday、hour-weekday 作为粗粒度样本上下文 |
| 训练策略强化 | 在模型与特征主线稳定后引入 pairwise loss、调度、EMA、平滑和正则 |

这些改动共同将公开榜 AUC 从官方 baseline 的约 `0.812` 提升到最终公开版本的 `0.828537`。其中，样本级摘要特征工程父线为 `0.816`，`v24` 在此基础上通过早期 query 条件化提升到 `0.821977`；`v87` 的主要新增收益来自训练策略强化，而不是替换 `v24-6` 的特征与模型路径。

## 信息流概览

一次前向计算可以概括为以下步骤。

| 步骤 | 说明 |
| --- | --- |
| 构造非序列 token | 将用户/物品离散特征压缩为 NS token，并加入用户 dense token 与物品 dense token |
| 编码行为序列 | 将每个行为域的 side-info 字段嵌入后投影到统一维度，并叠加时间 bucket embedding |
| 生成 query token | 对每个行为域做 masked mean pooling，并与非序列 token 拼接后生成该域的 query token |
| 行为域内解码 | query token 对各自行为域序列做 cross-attention，提取与当前样本相关的序列信息 |
| 跨域 token mixing | 将所有行为域 query token 与非序列 token 拼接，通过 RankMixer 做联合交互 |
| 输出预测 | 拼接最终 query token，投影为样本表示，加入元信息残差后输出 logit |

默认公开配置中，行为域数由数据 schema 决定；每个行为域生成 `num_queries=2` 个 query token，主干堆叠 `2` 个 HyFormer block，隐藏维度为 `d_model=64`。

## 非序列 token

非序列 token 包含四类信息：

1. 用户侧离散特征 token；
2. 用户侧 dense 表示 token；
3. 物品侧离散特征 token；
4. 物品侧 dense 表示 token。

离散特征默认使用 `RankMixerNSTokenizer`。该 tokenizer 会先按配置顺序嵌入离散字段，再将所有字段 embedding 拼接、分块并投影为固定数量的 NS token。公开配置中：

- 用户侧离散特征压缩为 `5` 个 NS token；
- 物品侧离散特征压缩为 `2` 个 NS token。

这一路径的重点是保持 token 数可控。后续实验显示，随意改变 token 数往往会牵动 `d_model`、RankMixer token mixing 约束和优化轨迹，因此不能把 token 数变化视为无害改动。

## 用户侧 dense 表示

用户侧 dense 特征没有被直接当作一个无结构向量处理。若运行时 schema 中存在预期字段布局，模型会启用结构化用户 dense 路径：

- 单独处理 dense 字段 `61`；
- 单独处理 dense 字段 `87`；
- 联合处理对齐字段组 `62-66` 的 dense 与对应用户离散 embedding；
- 联合处理对齐字段组 `89-91` 的 dense 与对应用户离散 embedding；
- 将上述组件融合回一个用户侧 dense token。

这一路径保留了 dense 字段内部结构，但不增加最终 token 数。该设计与后续负结果一致：把用户 dense 拆成多个额外 token、扩宽 hidden 维度或叠加更强 gate，都没有超过公开主线。

如果 schema 中缺少预期字段，代码会回退到普通的整体 dense 投影路径，以保证训练和评测行为可解释。

## 行为序列路径

每个行为域独立处理。序列中的各个 side-info 字段先分别 embedding，再拼接并投影到 `d_model` 维度。若启用时间 bucket，模型会额外加入 `time_embedding`。

默认公开配置保持较保守的序列设置：

- `seq_a=256`；
- `seq_b=256`；
- `seq_c=512`；
- `seq_d=512`；
- 默认不启用 RoPE；
- 默认不启用 causal sequence mask。

高基数字段会受到 `emb_skip_threshold` 与额外 embedding dropout 的约束。这样做的目的不是恢复所有可记忆 ID，而是减少高基数离散字段造成的过拟合风险。

## Query 生成与早期元信息条件化

每个行为域都会生成自己的 query token。生成过程如下：

1. 对该行为域序列 token 做 masked mean pooling；
2. 如启用早期样本级元信息条件化，将 `meta_feats` 投影后的上下文加到 pooled summary；
3. 将非序列 token 展平后与该域 summary 拼接；
4. 通过该行为域独立的 FFN 生成 `num_queries` 个 query token。

样本级摘要特征工程父线先验证了长度、截断、recency、span 和缺失模式等信息本身有效，公开 AUC 为 `0.816`。这一路径对应历史实验中的 `v24`，它进一步改变的是信息接入位置：样本级元信息不只在最终输出前做校正，而是提前参与“如何读取行为序列”的决策。

## HyFormer block

每个 HyFormer block 包含三个核心动作。

### 行为域内序列演化

每个行为域先经过独立的序列编码器。公开版本默认使用 transformer encoder 形式的 self-attention + FFN。

### Query 解码

每个行为域的 query token 对该域序列 token 做 cross-attention。该步骤让 query token 从对应行为域中提取与当前样本相关的信息。

### 跨域与静态上下文交互

所有行为域解码后的 query token 会与非序列 token 拼接，然后进入 RankMixer。RankMixer 在固定 token 表上做 token mixing 和逐 token FFN，使行为序列信息、用户/物品静态信息和 dense 信息发生联合交互。

在 `rank_mixer_mode='full'` 时，RankMixer 要求总 token 数 `T` 能整除 `d_model`。因此，改变 query 数、NS token 数或 dense token 数通常不是单点小改动，会影响整个主干几何。

## 样本级元信息路径

公开主线使用两处样本级元信息：

1. **早期 query 条件化**：`meta_feats` 投影后加到每个行为域的 pooled summary 上，影响 query token 生成。
2. **最终表示残差**：`meta_feats` 经过独立投影后，以 gated residual 形式加到最终样本表示上。

离散日历时间版本并没有新增第三条强时间分支，而是把 hour、weekday、hour-weekday one-hot 追加到已有 `meta_feats` 中，让它们继续通过上述两条路径进入模型。

这一点很重要：后续 direct time-logit、time representation residual、time query residual 和 Fourier 时间编码都未超过主线。更合理的解释是，时间信息在本任务中适合作为样本级上下文，而不是单独放大为强预测分支。

## 训练策略与最终公开版本

最终公开版本对应训练策略强化路线。它保留离散日历时间版本的模型结构和数据特征，主要调整训练侧策略：

- `bce_pairwise`；
- `pairwise_lambda=0.05`；
- cosine 学习率调度；
- `warmup_steps=500`；
- `EMA(decay=0.999)`；
- `label_smoothing=0.01`；
- dense 参数 `weight_decay=0.02`。

这说明公开榜 `0.828537` 的主要新增收益来自优化和正则化，而不是替换 `v24-6` 的模型路径。更准确的理解是：`v24-6` 提供有效的特征与结构主线，`v87` 在该主线上完成训练策略强化。

## 与官方 baseline 的关系

本仓库中所有分数解释均以官方 baseline 作为对照起点。官方 baseline 提供任务、数据格式、训练/评测口径和基本代码结构；本版本在此基础上整理出一条改进主线。

因此，阅读模型说明时应关注“哪些信息路径被增强”：

- 用户侧 dense 字段从直接投影转为结构化处理后融合；
- 样本级摘要特征工程先补回序列长度、截断、recency、span 和缺失模式等信息；
- 样本级摘要信息从末端补充转为同时参与 query 生成；
- 粗粒度日历时间通过已有元信息通道进入模型；
- 训练策略在保持特征与模型主线不变的条件下被系统强化。

## 设计边界

公开主线的负结果给出了一些明确边界：

- raw `user_id`、raw `item_id` 或 raw timestamp 不作为推荐输入路径；
- 用户 dense 多 token 展开加模型扩宽不是当前主线的推荐方向；
- 时间信号更适合走样本级元信息通道，而不是 direct logit 或额外强分支；
- 显式 user-item pair residual 在当前主线上风险较高；
- token 数、hidden 维度、dropout 和训练策略不应在同一实验中同时改变。

因此，公开模型的核心可以概括为：通过稳定的 token 几何组织静态与序列信息，通过样本级元信息弱条件化 query 生成，并在不扩大模型路径的前提下用训练策略提升最终结果。
