# StarCraft II micromanagement
Our code is modified from https://github.com/starry-sky6688/StarCraft.

This repository is our Pytorch implementations for Decomposed Soft Actor-Critic Method for Cooperative Multi-Agent Reinforcement Learning(https://arxiv.org/abs/2104.06655). 

In the paper, we incorporates the idea of the multi-agent value function decomposition and soft actor-critic framework effectively and proposes a new method mSAC. 
Experimental results demonstrate that mSAC significantly outperforms policy-based approach–COMA, and achieves competitive performance with SOTA value-based
approach–Qmix on most tasks in terms of asymptotic performance metric. In addition, mSAC has achieved significantly better results than Qmix in some tasks with large action spaces (such as 2c_vs_64zg, MMM2).

We trained these algorithms on [SMAC](https://github.com/oxwhirl/smac), the decentralised micromanagement scenario of [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty).

## Corresponding Papers

- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)

## Requirements

- python
- pytorch
- [SMAC](https://github.com/oxwhirl/smac)
- [pysc2](https://github.com/deepmind/pysc2)

## Acknowledgement

+ [SMAC](https://github.com/oxwhirl/smac)
+ [pymarl](https://github.com/oxwhirl/pymarl)
+ [starry-sky6688](https://github.com/starry-sky6688/StarCraft)

## Quick Start

```shell
$ python src/main.py --map=3m
```
To run different variant algorithms, you can change the first line in src/main.py [from runner_msac import Runner] to different runner_[alg. name]. 