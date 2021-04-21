# StarCraft
our code is modified from https://github.com/starry-sky6688/StarCraft

Our Pytorch implementations  for Decomposed Soft Actor-Critic Method for Cooperative Multi-Agent Reinforcement Learning(https://arxiv.org/abs/2104.06655). 
We incorporates the idea of the multi-agent value function decomposition and
soft actor-critic framework effectively.

We trained these algorithms on [SMAC](https://github.com/oxwhirl/smac), the decentralised micromanagement scenario of [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty).

## Corresponding Papers

- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)

## Requirements

- python
- torch
- [SMAC](https://github.com/oxwhirl/smac)
- [pysc2](https://github.com/deepmind/pysc2)

## Acknowledgement

+ [SMAC](https://github.com/oxwhirl/smac)
+ [pymarl](https://github.com/oxwhirl/pymarl)


## Quick Start

```shell
$ python src/main.py --map=3m --alg=qmix
```
