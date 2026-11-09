# 星际争霸 II 微操

[English](README.md) | [简体中文](README_zh-CN.md)

本仓库的代码基于 [starry-sky6688/StarCraft](https://github.com/starry-sky6688/StarCraft) 修改。

本仓库提供论文 [Decomposed Soft Actor-Critic Method for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2104.06655)（面向合作式多智能体强化学习的分解式软演员–评论家方法）的 PyTorch 实现。

论文将多智能体价值函数分解的思想与软演员–评论家（Soft Actor-Critic）框架相结合，提出了 mSAC 方法。实验结果表明，在大多数任务中，mSAC 的渐近性能显著优于基于策略的方法 COMA，并可与当时先进的基于价值的方法 QMIX 相媲美。此外，在一些动作空间较大的任务（如 `2c_vs_64zg`、`MMM2`）中，mSAC 取得了明显优于 QMIX 的结果。

我们在 [SMAC](https://github.com/oxwhirl/smac) 环境中训练这些算法。SMAC 基于[星际争霸 II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty)的分散式微操场景构建。

## 相关论文

- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)

## 环境依赖

- Python
- PyTorch
- [SMAC](https://github.com/oxwhirl/smac)
- [pysc2](https://github.com/deepmind/pysc2)

## 致谢

- [SMAC](https://github.com/oxwhirl/smac)
- [pymarl](https://github.com/oxwhirl/pymarl)
- [starry-sky6688](https://github.com/starry-sky6688/StarCraft)

## 快速开始

```shell
python src/main.py --map=3m
```

如需运行其他算法变体，可将 `src/main.py` 中的 `from runner_msac import Runner` 改为从相应的 `runner_<算法名称>` 模块导入 `Runner`。

## 许可证

本仓库采用 [MIT License](LICENSE)。欢迎大家继续分析、使用和改进我们的实现。

## 引用

如果我们的[论文](https://arxiv.org/abs/2104.06655)和本仓库对你的研究有所帮助，请考虑引用：

```bibtex
@article{pu2021decomposed,
  title={Decomposed soft actor-critic method for cooperative multi-agent reinforcement learning},
  author={Pu, Yuan and Wang, Shaochen and Yang, Rui and Yao, Xin and Li, Bin},
  journal={arXiv preprint arXiv:2104.06655},
  year={2021}
}
```
