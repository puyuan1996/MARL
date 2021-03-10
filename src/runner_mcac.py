import numpy as np
import os
from common.rollout import RolloutWorker, CommRolloutWorker
# from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
# from pg.basic_controller import MAgents
# from pg.qmix_ac_sample import QMIX_PG  # TODO
# from pg.qmix_ac_total import QMIX_PG  # TODO
# from pg.qmix_ac_total_counterfactual import QMIX_PG  # TODO
# from pg.qmix_ac_local-total_counterfactual import QMIX_PG  # TODO
# from pg.qmix_ac_local-total import QMIX_PG  # TODO
# from agent.agent_pg import Agents
# from pg_softmax.qmix_ac_total import QMIX_PG  # TODO
from ac_discrete.qmix_ac_total_counterfactual import QMIX_PG  # TODO
# from pg_softmax.qmix_sac import QMIX_PG  # TODO
from agent.agent_mcac import Agents
# from pg.qmix_ddpg import QMIX_PG  # TODO
# from agent.agent_pg_ddpg import Agents

class Runner:
    def __init__(self, env, args):
        self.env = env
        self.args=args

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            # self.agents = Agents(args)
            self.agents = Agents(args)
            self.qmix_pg_learner = QMIX_PG(self.agents, args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if args.learn and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.critic_buffer = ReplayBuffer(args, args.critic_buffer_size)
            self.actor_buffer = ReplayBuffer(args, args.actor_buffer_size)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        # self.save_path = self.args.result_dir + '/'  +'pg_gumbel_softmax_logpi' + '/' + args.alg  +'/'+ args.map
        # self.save_path = self.args.result_dir + '/'  +'ac_sample1' + '/' + args.alg  +'/'+ args.map
        # self.save_path = self.args.result_dir + '_my' + '/' + 'ac' + '/' + args.alg + '/' + args.map
        # self.save_path = self.args.result_dir + '_my' + '/' + 'ac_qtotal' + '/' + args.alg + '/' + args.map
        # tmp = f'clamp2-5_advnorm_entropy_'+f'{args.actor_buffer_size}_{args.critic_buffer_size}_{args.actor_train_steps}_{args.critic_train_steps}_' \
        #                                    f'{args.actor_update_delay}_{args.critic_lr}_{args.loss_coeff_entropy}'

        # tmp =f'{args.anneal_epsilon}_'+ f'{args.actor_buffer_size}_{args.critic_buffer_size}_{args.actor_train_steps}_{args.critic_train_steps}_' \
        #                                      f'{args.actor_update_delay}_{args.critic_lr}'
        # self.save_path = self.args.result_dir + '_qmix_ddpg' + '/' + tmp + '/'+args.alg + '/' + args.map


        tmp = f'clamp2-5_' + f'{args.loss_coeff_entropy}_'+f'{args.buffer_size}_{args.actor_buffer_size}_{args.critic_buffer_size}_{args.actor_train_steps}_{args.critic_train_steps}_' \
                             f'{args.actor_update_delay}_{args.critic_lr}'  # f'clamp2-5_'+  anneal_epsilon
        self.save_path = self.args.result_dir+'/linear_mix/' + 'qmix_ac_total_cf' + '/' + tmp + '/' + args.map

        # tmp = f'seq-agent-update_clamp2-5_advnorm_{args.anneal_epsilon}_'+f'{args.actor_buffer_size}_{args.critic_buffer_size}_{args.actor_train_steps}_{args.critic_train_steps}_{args.actor_update_delay}_{args.critic_lr}'

        # tmp = f'clamp2-5_advnorm_'+f'{args.actor_buffer_size}_{args.critic_buffer_size}_{args.actor_train_steps}_{args.critic_train_steps}_{args.actor_update_delay}_{args.critic_lr}'
        # self.save_path = 'onpolicy/' + self.args.result_dir + '_my_softmax_epsilon' + '/' + 'ac_total' + '/' + tmp + '/' + args.alg + '/' + args.map

        # tmp = f'clamp0-1' + f'{args.actor_buffer_size}_{args.critic_buffer_size}_{args.actor_train_steps}_{args.critic_train_steps}_{args.actor_update_delay}_{args.critic_lr}'
        # self.save_path = 'qmix-sac/'  + '/' + tmp + '/' + args.alg + '/' + args.map


        # self.save_path = 'test/' + args.alg + '/' + args.map

        # tmp = '32_2_1_1e-4'
        # self.save_path = self.args.result_dir  + '/' + 'ac_sample' + '/' + tmp + '/' + args.alg + '/' + args.map
        # self.save_path = self.args.result_dir  + '/' + 'ac_sample_evalmax' + '/' + tmp + '/' + args.alg + '/' + args.map
        # self.save_path = self.args.result_dir  + '/' + 'ac_total' + '/' + tmp + '/' + args.alg + '/' + args.map
        # self.save_path = self.args.result_dir + '_softmax_evalmax_epsilon' + '/' + 'ac_total'+ '/' + tmp + '/' + args.alg + '/' + args.map


        # --map = 3m - -alg = qmix --actor_buffer_size=5000  --critic_buffer_size=5000  --cuda=True
        # self.save_path = self.args.result_dir + '_my' + '/' + 'ac_local-total' + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        train_steps = 0
        epsilon=self.args.epsilon#初始epsilon
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
            # epsilon = 0 if evaluate else self.epsilon
            # if self.args.epsilon_anneal_scale == 'episode':
            #     epsilon = epsilon - self.args.anneal_epsilon if epsilon > self.args.min_epsilon else epsilon

            if self.args.epsilon_anneal_scale == 'epoch':
                epsilon = epsilon - self.args.anneal_epsilon if epsilon > self.args.min_epsilon else epsilon
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):  # 1
                episode, _, _ = self.rolloutWorker.generate_episode(episode_idx,evaluate=False,epsilon=epsilon)
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
                self.actor_buffer.store_episode(episode_batch)
                # if epoch % 16 == 0:  # 2
                for train_step in range(self.args.critic_train_steps):  # 1  # 16
                    mini_batch = self.critic_buffer.sample( min(self.critic_buffer.current_size, self.args.critic_batch_size))  # 32 episodes # 16
                    self.qmix_pg_learner.train_critic(mini_batch, self.args.episode_limit, train_steps)
                    train_steps += 1
                if epoch % self.args.actor_update_delay == 0:  # 2
                    for train_step in range(self.args.actor_train_steps):  # 1 # 16
                        mini_batch = self.actor_buffer.sample( min(self.actor_buffer.current_size, self.args.actor_batch_size))  # 16 episodes  # 16
                        self.qmix_pg_learner.train_actor(mini_batch, self.args.episode_limit)
        self.plt(num)

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag = self.rolloutWorker.generate_episode(epoch, evaluate=True,epsilon=0)
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
