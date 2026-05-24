# 参考资料

公开方案并不是对某一篇论文或某一个外部仓库的直接复刻。比赛期间，论文、外部仓库和本地实验结果共同用于判断哪些方向值得验证。

## 论文参考

### 推荐系统与 CTR/CVR 建模

- Covington et al., *Deep Neural Networks for YouTube Recommendations*  
  [https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)

- Guo et al., *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*  
  [https://arxiv.org/abs/1703.04247](https://arxiv.org/abs/1703.04247)

- Naumov et al., *Deep Learning Recommendation Model for Personalization and Recommendation Systems*  
  [https://arxiv.org/abs/1906.00091](https://arxiv.org/abs/1906.00091)

### 兴趣建模与 target-aware 思路

- Zhou et al., *Deep Interest Network for Click-Through Rate Prediction*  
  [https://arxiv.org/abs/1706.06978](https://arxiv.org/abs/1706.06978)

DIN 相关工作主要作为对照参考。当前主线中的 DIN 式候选感知 query 变体没有形成稳定收益。

### 表格与序列建模

- Huang et al., *TabTransformer: Tabular Data Modeling Using Contextual Embeddings*  
  [https://arxiv.org/abs/2012.06678](https://arxiv.org/abs/2012.06678)

- Zhai et al., *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*  
  [https://arxiv.org/abs/2402.17152](https://arxiv.org/abs/2402.17152)

HSTU 相关工作主要提供序列建模思路背景，并不是公开主线的直接来源。

## 比赛期间参考过的外部仓库

- Puiching-Memory/TAAC_2026  
  [https://github.com/Puiching-Memory/TAAC_2026](https://github.com/Puiching-Memory/TAAC_2026)

- WzH1learner/taac-codes  
  [https://github.com/WzH1learner/taac-codes](https://github.com/WzH1learner/taac-codes)

- axdyer/TAAC-2026-LeaderBoard  
  [https://github.com/axdyer/TAAC-2026-LeaderBoard](https://github.com/axdyer/TAAC-2026-LeaderBoard)

- denghuigeng/taac  
  [https://github.com/denghuigeng/taac](https://github.com/denghuigeng/taac)

- W-void/TAAC  
  [https://github.com/W-void/TAAC](https://github.com/W-void/TAAC)

- yst09/TAAC2026  
  [https://github.com/yst09/TAAC2026](https://github.com/yst09/TAAC2026)

- ralgond/TAAC2026  
  [https://github.com/ralgond/TAAC2026](https://github.com/ralgond/TAAC2026)

- fireflycsq/ad_alg  
  [https://github.com/fireflycsq/ad_alg](https://github.com/fireflycsq/ad_alg)

## 参考资料的作用

这些资料主要用于三类判断：

1. 识别值得验证的方向，例如训练策略、用户侧 dense 结构和序列局部性；
2. 判断某些方法迁移到当前代码框架后是否稳定；
3. 解释本地负结果和外部材料之间的共性。

最终公开主线仍以本仓库中的代码、公开榜记录和实验笔记为准。
