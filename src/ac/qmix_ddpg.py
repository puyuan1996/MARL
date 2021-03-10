import torch
import os
from network.base_net_ddpg import RNN
from network.qmix_net_ddpg import QMixNet
from .misc import gumbel_softmax, onehot_from_logits
import copy
import numpy as np


class QMIX_PG():
    def __init__(self, agent, args):
        self.agent = agent
        self.target_policy = copy.deepcopy(self.agent.policy)  # TODO

        self.policy_params = list(agent.policy.parameters())
        self.policy_optimiser = torch.optim.RMSprop(params=self.policy_params,
                                                    lr=args.actor_lr)  # , alpha=args.optim_alpha, eps=args.optim_eps)
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents
        input_shape += self.n_actions  # todo action作为q网络输入
        # 神经网络
        self.eval_rnn = RNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet(args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
            self.agent.policy.cuda()
            self.target_policy.cuda()
        # tmp = f'{args.actor_buffer_size}_{args.actor_update_delay}_{args.actor_train_steps}_{args.critic_lr}'
        tmp = f'{args.actor_buffer_size}_{args.critic_buffer_size}_{args.actor_train_steps}_{args.critic_train_steps}_' \
              f'{args.actor_update_delay}_{args.critic_lr}'  # f'clamp2-5'+

        self.model_dir = args.model_dir + '/' + 'qmix_ddpg' + '/' + tmp + '/' + args.alg + '/' + args.map  # TODO
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                path_policy = self.model_dir + '/policy_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                self.target_policy.load_state_dict(torch.load(path_policy, map_location=map_location))  # TODO
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        self.target_policy.load_state_dict(self.agent.policy.state_dict())  # TODO

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.critic_lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg QMIX')

    def train_actor(self, batch, max_episode_len, update_agent_id=0):  # EpisodeBatch
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'], batch['avail_u'], batch['avail_u_next'], \
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        # q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        policy_outs = []
        # import numpy as np
        # i = np.random.randint(0, self.n_agents)#todo
        # rand_agent_id = np.random.randint(0, self.n_agents)  # todo  # # TODO
        i = update_agent_id + 1 if update_agent_id < self.n_agents - 1 else 0
        for transition_idx in range(max_episode_len):
            # inputs_buffer_a, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            inputs, inputs_next = self._get_inputs_original(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            agent_outs, self.policy_hidden = self.agent.policy(inputs, self.policy_hidden)

            a_i_grad_negi_detached = []  # TODO 每次只有一个agent的输出动作传回梯度
            # agent_outs_detached = agent_outs.detach()
            # for i in range(self.n_agents):
            # agent_outs_temp = copy.deepcopy(agent_outs_detached)
            # agent_outs_temp[i] = agent_outs[i]
            agent_outs_temp = torch.cat((agent_outs[..., :i].detach(), agent_outs[..., i:i + 1], agent_outs[..., i + 1:].detach()), dim=-1)
            # a_i_grad_negi_detached.append(agent_outs_temp)

            # policy_out=agent_outs.clone()
            # policy_outs.append(policy_out)
            # agent_outs, self.target_policy_hidden = self.target_policy(inputs_next, self.target_policy_hidden)
            avail_actions_ = avail_u[:, transition_idx]
            reshaped_avail_actions = avail_actions_.reshape(episode_num * self.n_agents, -1)
            agent_outs_temp[reshaped_avail_actions == 0] = -1e11
            # agent_outs = gumbel_softmax(agent_outs, hard=True)  # one-hot TODO ac_sample1
            # agent_outs = agent_outs.view(episode_num, self.n_agents, -1) # (1,3,9)
            # action_next = agent_outs.max(dim=2, keepdim=True)[1] # (1,3,1)
            # # action_next = torch.nonzero(agent_outs).squeeze()
            # actions_next_sample.append(action_next) # 选择动作的序号

            agent_outs_temp = gumbel_softmax(agent_outs_temp, hard=True)  # one-hot TODO 采样的离散动作
            agent_outs_temp = agent_outs_temp.view(episode_num * self.n_agents, -1)

            inputs_pi_a = torch.cat([inputs, agent_outs_temp], dim=-1)

            q_eval, self.eval_hidden = self.eval_rnn(inputs_pi_a, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            # q_target, self.target_hidden = self.target_rnn(inputs_next_pi_a, self.target_hidden)

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)  # 把q_eval维度重新变回(8, 5,1)
            # q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            # q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        # q_targets = torch.stack(q_targets, dim=1)
        # policy_outs = torch.stack(policy_outs, dim=1)

        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        # q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_evals = q_evals.squeeze(3)
        q_total_eval = self.eval_qmix_net(q_evals, s)  # [1, 60, 1]

        policy_loss = - ((q_total_eval.view(-1)) * mask).sum() / mask.sum()
        # policy_loss += (policy_outs ** 2).mean() * 1e-3

        # Optimise agents
        self.policy_optimiser.zero_grad()
        policy_loss.backward()  # policy gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_params, self.args.grad_norm_clip)
        self.policy_optimiser.step()

        # compute parameters sum for debugging
        p_sum = 0.
        for p in self.policy_params:
            p_sum += p.data.abs().sum().item() / 100.0

        self.soft_update()

    def train_critic(self, batch, max_episode_len, train_step,
                     epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'], batch['avail_u'], batch['avail_u_next'], \
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        # q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs_buffer_a, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id

            # inputs, inputs_next = self._get_inputs_original(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs_buffer_a = inputs_buffer_a.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            agent_outs, self.target_policy_hidden = self.target_policy(inputs_next, self.target_policy_hidden)
            avail_actions_ = avail_u[:, transition_idx]
            reshaped_avail_actions = avail_actions_.reshape(episode_num * self.n_agents, -1)
            agent_outs[reshaped_avail_actions == 0] = -1e11

            # agent_outs = gumbel_softmax(agent_outs, hard=True)  # one-hot TODO ac_sample1
            # agent_outs = agent_outs.view(episode_num, self.n_agents, -1) # (1,3,9)
            # action_next = agent_outs.max(dim=2, keepdim=True)[1] # (1,3,1)
            # # action_next = torch.nonzero(agent_outs).squeeze()
            # actions_next_sample.append(action_next) # 选择动作的序号

            agent_outs = onehot_from_logits(agent_outs)  # one-hot TODO 采样的离散动作
            # agent_outs = agent_outs.view(episode_num, self.n_agents, -1)
            agent_outs = agent_outs.view(episode_num * self.n_agents, -1)

            inputs_next_pi_a = torch.cat([inputs_next, agent_outs], dim=-1)

            q_eval, self.eval_hidden = self.eval_rnn(inputs_buffer_a,
                                                     self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target, self.target_hidden = self.target_rnn(inputs_next_pi_a, self.target_hidden)

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)  # 把q_eval维度重新变回(8, 5,1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)

        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        # q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_evals_buffer_a = q_evals.squeeze(3)

        episode_num = batch['o'].shape[0]

        # pi = actions_prob.view(-1, self.n_actions)
        # Calculated baseline
        q_targets_pi_a = q_targets.view(episode_num, max_episode_len, -1).detach()  # (1,60,3)

        q_total_eval = self.eval_qmix_net(q_evals_buffer_a, s)  # [1, 60, 1]
        q_total_target = self.target_qmix_net(q_targets_pi_a, s_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        # if train_step > 0 and train_step % self.args.target_update_cycle == 0: # 200
        #     self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        #     self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        #     self.target_policy.load_state_dict(self.agent.policy.state_dict()) #TODO
        # Update the frozen target models

    def _get_inputs_original(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号 30+9+3

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next  # (3,42)

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号 30+9+3

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs.append(u_onehot[:, transition_idx])  # todo 加上a_t
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next  # (3,42)

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.policy_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_policy_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))  # TODO

        if self.args.cuda:
            self.eval_hidden = self.eval_hidden.cuda()
            self.target_hidden = self.target_hidden.cuda()
            self.policy_hidden = self.policy_hidden.cuda()
            self.target_policy_hidden = self.target_policy_hidden.cuda()

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.model_dir + '/' + num + '_rnn_net_params.pkl')
        torch.save(self.agent.policy.state_dict(), self.model_dir + '/' + num + '_policy_net_params.pkl')

    def soft_update(self):
        self.tau = 0.005
        for param, target_param in zip(self.eval_rnn.parameters(), self.target_rnn.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.eval_qmix_net.parameters(), self.target_qmix_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.agent.policy.parameters(), self.target_policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
