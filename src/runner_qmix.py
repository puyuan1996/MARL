#-*-coding:UTF-8-*-
import numpy as np
import os
from common.rollout_qmix import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
# from pg.basic_controller import MAgents
from policy.qmix import QMIX  # TODO
# from agent.agent_pg import Agents


class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            # self.agents = Agents(args)
            self.agents = Agents(args)

            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if args.learn and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find(
                'reinforce') == -1:  # these 3 algorithms are on-poliy
            self.critic_buffer = ReplayBuffer(args, args.critic_buffer_size)
            # self.actor_buffer = ReplayBuffer(args, args.actor_buffer_size)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        # self.save_path = self.args.result_dir + '/'  +'pg_gumbel_softmax_logpi' + '/' + args.alg  +'/'+ args.map
        # self.save_path = self.args.result_dir + '/'  +'ac_sample1' + '/' + args.alg  +'/'+ args.map
        # self.save_path = self.args.result_dir + '_my' + '/'+'hard_update'+'/' + args.alg + '/' + args.map
        self.save_path = self.args.result_dir + '_my' + '/'+'soft_update'+'/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        train_steps = 0
        # print('Run {} start'.format(num))
        for epoch in range(self.args.n_epoch):
            print('Run {}, train epoch {}'.format(num, epoch))
            if epoch % self.args.evaluate_cycle == 0:  # 100
                win_rate, episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.plt(num)

            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):  # 1
                episode, _, _ = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find(
                    'reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.critic_buffer.store_episode(episode_batch)
                # self.actor_buffer.store_episode(episode_batch)
                for train_step in range(self.args.critic_train_steps):  # 1
                    mini_batch = self.critic_buffer.sample(min(self.critic_buffer.current_size, self.args.critic_batch_size))  # 32 episodes
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
                # if epoch % self.args.actor_update_delay == 0:  # 2
                #     for train_step in range(self.args.actor_train_steps):  # 1
                #         mini_batch = self.actor_buffer.sample(
                #             min(self.actor_buffer.current_size, self.args.actor_batch_size))  # 16 episodes
                #         self.qmix_pg_learner.train_actor(mini_batch, self.args.episode_limit)
        self.plt(num)

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        # plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rate')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        # plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.xlabel('episodes*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
