# 实验目录

本文档在 [experiment_summary.md](./experiment_summary.md) 的基础上记录更多实验方向。它不是逐条实验日志，而是对有代表性的正向和负向证据进行整理。

## 主线有效方向

| 方向 | 公开结果 | 主要观察 |
| --- | ---: | --- |
| 样本级摘要特征工程 | `0.816` | 长度、截断、recency、span 和缺失模式等 meta-summary 特征能够补充固定长度序列表示 |
| 样本级元信息提前参与 query 生成 | `0.821977` | 元信息提前参与表示提取，比只在分类头附近使用更有效 |
| 样本级离散日历时间 | `0.825905` | hour、weekday、hour-weekday 这类粗粒度上下文在当前框架中稳定 |
| 训练策略强化 | `0.828537` | 保持特征与模型主线不变，训练策略带来后期主要提升 |

## 未形成主线的方向

| 方向 | 公开结果 | 观察 |
| --- | ---: | --- |
| direct time-logit 校准 | `0.823352` | 直接时间校准过强，排序表现不稳定 |
| contextual NS gating | `0.823248` | 全局 gate 没有形成稳定收益 |
| 显式 user-item pair 残差 | `0.818371` | 强 pair 路线明显扰动当前表示 |
| 用户 dense heavy-tail 简单缩放 | `0.823468` | 不是当前主要瓶颈 |
| 拉长序列长度 | `0.821260` | 更长序列没有自然带来更高公开分 |
| Fourier 时间编码 | `0.820367` | 表达能力更强，但不如离散粗时间稳定 |
| 用户 dense 拆成多个 token | `0.820980` | 破坏主线 token 几何 |
| gate + 更大 dropout | `0.817980` | 增加优化负担，未形成收益 |
| DIN 式候选感知 query | `0.817735` | 在当前 HyFormer/RankMixer 主线上效果较弱 |
| item exposure / hotness / recency sidecar | `0.823919` | item 侧统计有信号，但该接法未超过主线 |
| refined semantic tokenizer / ordering | `0.820276` | 手工语义重排未形成稳定改进 |
| tiny late cross head | `0.822906` | 小型末端交叉不足以超过主线 |
| tiny user-dense output mask | `0.824370` | 接近但仍低于主线 |
| 小型 item-local delta | `0.822528` | item 字段重要，但额外 item-local 分支未带来稳定收益 |
| hash fallback 对照 | `0.827938` | 仍强于旧主线，但低于训练策略强化版本 |

## 共同现象

### 主线对 token 几何较敏感

token 数量、query 位置、时间信号接入方式和 target-aware 路线的改变，都会让公开分出现明显波动。用户 dense 多 token、DIN query 和显式 pair residual 的结果都支持这一点。

### 更强表达能力不必然带来更高公开分

许多更复杂的设计没有转化为更高分数，例如：

- 更多 token；
- 更长序列；
- 更复杂时间编码；
- 显式 pair interaction；
- 更强 target-aware 分支。

因此，公开主线更强调稳定的信息接入方式，而不是持续增加结构复杂度。

### valid 和 public 可能不一致

部分分支在训练 valid 上并不差，但公开榜没有对应收益。后续实验记录因此更重视：

- 对照设置是否清楚；
- 公开结果是否可复核；
- 负结果是否完整记录；
- 是否只改变一个核心变量。

## 后期探索方向

### BCE、focal 及更激进 focal

后期系统测试过从 `bce_pairwise` 回退到纯 BCE，以及 focal 系列参数。它们会改变训练轨迹和最佳 epoch 位置，但未形成稳定优于公开主线的结果。

### semi-local causal mask

序列 self-attention 中曾测试 semi-local causal mask 以及 bottom-full top-local 变体。这类实验在训练曲线上并非完全无效，但公开结果不足以替代主线。

### 用户 embedding 分离与额外 pair

将对齐的用户 int/dense 字段先分离，再加入额外 pair 路线的实验没有形成稳定收益。

### grouped user dense

更显式的 grouped user dense projector 说明用户侧 dense 内部结构值得处理，但在当前主线上并未超过训练策略强化版本。

## 保留实验目录的价值

这些记录的价值不只在于说明什么有效，也在于界定边界：

1. 样本级摘要特征工程本身有效。
2. 样本级元信息提前进入模型是有效方向。
3. 粗粒度时间上下文比更强时间建模稳定。
4. 当前主线的 token 几何较脆弱。
5. pair、DIN 和复杂 target-aware 路线在本框架中没有形成稳定答案。
6. 主干结构稳定后，训练策略可能比继续加结构更重要。
