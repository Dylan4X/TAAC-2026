# 负结果整理

本文档记录未成为公开主线的代表性方向。以下结论只对应本次比赛数据、代码框架和验证口径，不应被理解为对相关方法的普遍否定。

## 显式 pair 路线不稳定

多次实验尝试引入更显式的 user-item pair 建模，包括：

- 粗粒度 user-item pair residual；
- DIN 式候选感知 query；
- 用户侧 int/dense 分离后再加入额外 pair 路线。

这些分支的共同现象是：

- 更容易扰动 token 几何；
- 训练后期稳定性变差；
- valid 与 public 的对应关系变弱。

这说明在当前主线中，显式加强 pair 交互并不一定优于保持原有聚合路径。

## 用户侧 dense 信号需要结构化，而不是扩大暴露面

用户侧 dense 特征很重要，但简单扩大其暴露面没有稳定收益。

测试过的变体包括：

- 多 token 展开；
- hidden 维度扩宽；
- 更大 dropout；
- 输出侧 mask；
- grouped projector 迁移实现。

这些结果共同说明：用户侧 dense 内部结构需要处理，但更适合在单个保守 token 内完成，而不是继续增加 token 数或额外分支。

## 更强时间建模没有延续收益

粗粒度日历时间在主线中有效，但更强时间建模没有继续提升。

测试过的方向包括：

- direct time-logit；
- Fourier 时间编码；
- time representation residual；
- time query residual。

在当前主线中，时间信息更适合作为样本级上下文信号，而不是被放大为独立强分支。

## gate 和 residual 不是通用修复器

多个实验本质上是在已有表示上增加 gate、residual、mask 或小型 cross 模块：

- contextual NS gating；
- dense component gate；
- late cross head；
- output-side user dense mask。

这些设计局部上合理，但没有形成稳定主线。它们通常会增加优化复杂度，却不能保证带来足够可靠的信息增量。

## valid 表现不能单独决定方向

部分分支在 valid 上表现不差，但公开榜没有对应提升。这类现象常见于：

- pair 路线；
- 更复杂时间编码；
- 显式 target-aware 路线；
- 更强 dense 暴露；
- item hotness / recency 统计分支。

因此，最终实验记录更强调公开结果、对照变量和负结果背景，而不是只看单次 valid 曲线。

## 对公开主线的约束

这些负结果对公开主线形成了三个约束：

1. 保持 token 几何保守。
2. 优先改进信息接入位置，而不是叠加分支。
3. 在模型主干稳定后，优先验证训练策略，而不是继续扩大结构复杂度。
