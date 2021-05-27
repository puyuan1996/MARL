import torch
import os
from network.base_net import RNN
from network.qmix_net_linear import QMixNet  # todo
from .misc import gumbel_softmax
import copy
import torch.nn.functional as F

# torch.cuda.set_device(1)  # id=0, 1, 2 ,4等
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
        tmp = f'clamp2-5_' + f'{args.loss_coeff_entropy}_' + f'{args.actor_buffer_size}_{args.critic_buffer_size}_{args.actor_train_steps}_{args.critic_train_steps}_' \
                                                             f'{args.actor_update_delay}_{args.critic_lr}'  # f'{args.anneal_epsilon}_'
        self.model_dir = 'linear_mix/' + args.model_dir + '/qmix_ac_total_counterfactual' + '/' + tmp + '/' + args.map

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
        # Get the relevant quantities
        # bs = batch.batch_size
        # max_t = batch.max_seq_length
        # actions = batch["u"][:, :-1]
        # terminated = batch["terminated"][:, :-1].float()
        # avail_actions = batch["avail_u"][:, :-1]
        # # mask = batch["filled"][:, :-1].float()
        # mask = 1 - batch["padded"].float()
        # mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # mask = mask.repeat(1, 1, self.n_agents).view(-1)
        # states = batch["s"][:, :-1]

        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        # s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
        #                                                      batch['r'], batch['avail_u'], batch['avail_u_next'], \
        #                                                      batch['terminated']
        s = batch['s']
        terminated = batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        # actions = batch["u"][:, :-1]
        actions = batch["u"]
        avail_u = batch["avail_u"]
        # terminated = batch["terminated"][:, :-1].float()
        # avail_actions = batch["avail_u"][:, :-1]

        if self.args.cuda:
            s = s.cuda()
            # u = u.cuda()
            # r = r.cuda()
            # s_next = s_next.cuda()
            # terminated = terminated.cuda()
            actions = actions.cuda()
            avail_u = avail_u.cuda()
            mask = mask.cuda()

        # # build q
        # inputs = self.critic._build_inputs(batch, bs, max_t)
        # q_vals = self.critic.forward(inputs).detach()[:, :-1]

        # episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []

        actions_prob = []
        # self.agent.init_hidden(batch.batch_size)
        # self.policy_hidden= self.policy.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs,
                                                     self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions) Q_i
            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)

            agent_outs, self.policy_hidden = self.agent.policy(inputs, self.policy_hidden)
            avail_actions_ = avail_u[:, transition_idx]
            reshaped_avail_actions = avail_actions_.reshape(episode_num * self.n_agents, -1)
            agent_outs[reshaped_avail_actions == 0] = -1e11
            # agent_outs = torch.nn.functional.softmax(agent_outs, dim=-1)
            # agent_outs = gumbel_softmax(agent_outs, hard=False) #todo
            agent_outs = F.softmax(agent_outs / 1, dim=1)  # 概率分布

            # If hard=True, then the returned sample will be one-hot, otherwise it will
            # be a probabilitiy distribution that sums to 1 across classes
            agent_outs = agent_outs.view(episode_num, self.n_agents, -1)
            actions_prob.append(agent_outs)  # 每个动作的概率

        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_vals = torch.stack(q_evals, dim=1)

        # mac_out = []
        # self.mac.init_hidden(batch.batch_size)
        # for t in range(batch.max_seq_length - 1):
        #     agent_outs = self.mac.forward(batch, t=t)
        #     mac_out.append(agent_outs)
        actions_prob = torch.stack(actions_prob, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        # actions_prob[avail_actions == 0] = 0
        # actions_prob = actions_prob / actions_prob.sum(dim=-1, keepdim=True)
        # actions_prob[avail_actions == 0] = 0

        # Calculated baseline
        q_taken_actions = torch.gather(q_vals, dim=3, index=actions).squeeze(3)  # (1,60,3)  TODO NOTE
        q_taken = self.eval_qmix_net(q_taken_actions, s).detach()  # TODO # (1,60,1)
        Q_taken = q_taken.repeat(repeats=(1, 1, self.n_agents))  # (1,60,3)

        pi = actions_prob.view(-1, self.n_actions)
        # q_targets_mean = torch.sum(actions_prob * q_vals, dim=-1).view(-1).detach() # (180)
        q_mean_actions = torch.sum(actions_prob * q_vals, dim=-1).detach()  # (1,60,3)
        q_i_mean_negi_taken = []
        # torch.stack( [torch.cat((q_targets_mean[:, :, i].unsqueeze(2), torch.cat((q_taken[:, :, :i], q_taken[:, :, i + 1:]), dim=2)),
        # dim=2) for i in range(3)],dim=1) # 顺序不对
        q_taken_actions_detached = q_taken_actions.detach()
        for i in range(self.n_agents):
            q_temp = copy.deepcopy(q_taken_actions_detached)
            q_temp[:, :, i] = q_mean_actions[:, :, i]  # q_i(mean_action) q_-i(tacken_action),
            q_i_mean_negi_taken.append(q_temp)
        q_i_mean_negi_taken = torch.stack(q_i_mean_negi_taken, dim=2)  # [1, 60, 3, 3]
        q_i_mean_negi_taken = q_i_mean_negi_taken.view(episode_num, -1, self.n_agents)  # [1, 60*3, 3]
        # TODO
        # s_repeat = s.repeat(repeats=(1, self.n_agents, 1))  # (1,60,48)-> # (1,60*3,48) #TODO 顺序错误
        s_repeat = s.repeat(repeats=(1, 1, self.n_agents))  # (1,60,48)-> # (1,60,48*3)
        s_repeat = s_repeat.view(episode_num, self.n_agents, -1)  # (1,60,48*3)-> # (1,60*3,48)

        Q_i_mean_negi_taken = self.eval_qmix_net(q_i_mean_negi_taken, s_repeat).detach()  # TODO #  (1,60*3,1)
        # q_total_target = q_total_target.repeat(repeats=(1, 1, self.n_agents)) # (1,60,3)
        Q_i_mean_negi_taken = Q_i_mean_negi_taken.view(episode_num, -1, self.n_agents)  # (1,60*3,1)->(1,60,3)

        # Calculate policy grad with mask
        pi_taken = torch.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)

        # # TODO normalnize
        # advantages_original=(q_taken.view(-1,self.n_agents)-baseline.view(-1,self.n_agents))
        # advantages = (advantages_original - advantages_original.mean(dim=0)) / advantages_original.std(dim=0) #
        # # advantages = (advantages_original-advantages_original.mean(dim=1).repeat(repeats=(1, max_episode_len, 1)))/advantages_original.std(dim=1).repeat(repeats=(1, max_episode_len, 1))
        # advantages=advantages.view(-1).detach()

        # TODO no normalnize
        advantages = (Q_taken.view(-1) - Q_i_mean_negi_taken.view(-1)).detach()
        # advantages = (Q_taken.view(-1) ).detach()#TODO

        if self.args.advantage_norm:  # TODO
            EPS = 1e-10
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
            # advantages = (q_taken.view(-1) - baseline.view(-1)).detach()
            # advantages = (-log_pi_taken + q_taken.view(-1).detach() - baseline) # TODO
        # import numpy as np
        # update_agent_id = np.random.randint(0, self.n_agents)  # TODO
        # update_agent_id=update_agent_id+1 if update_agent_id<self.n_agents-1 else 0
        # policy_loss = - ((advantages[update_agent_id::self.n_agents] * log_pi_taken[update_agent_id::self.n_agents]) * mask[update_agent_id::self.n_agents]).sum() / mask[update_agent_id::self.n_agents].sum()
        policy_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()

        # policy_entropy = torch.mean(torch.exp(log_pi_taken) * log_pi_taken)
        # regu_policy_loss = policy_loss + self.args.loss_coeff_entropy * policy_entropy

        # Optimise agents
        self.policy_optimiser.zero_grad()
        policy_loss.backward()  # policy gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_params, self.args.grad_norm_clip)
        self.policy_optimiser.step()
        self.soft_update()

        if policy_loss.item() > 1e3:
            print("policy_loss:", policy_loss.item())

    def train_critic(self, batch, max_episode_len, train_step):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
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
        r = (r - r.mean()) / (r.std() + 1e-6)  # todo
        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)

        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        episode_num = batch['o'].shape[0]
        q_targets_sample = []

        actions_prob = []
        actions_next_sample = []
        # self.agent.init_hidden(batch.batch_size)
        # self.policy_hidden= self.policy.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
            # agent_outs, self.target_policy_hidden = self.target_policy(inputs_next, self.target_policy_hidden) TODO
            agent_outs, self.policy_hidden = self.agent.policy(inputs_next, self.policy_hidden)

            avail_actions_ = avail_u[:, transition_idx]
            reshaped_avail_actions = avail_actions_.reshape(episode_num * self.n_agents, -1)
            agent_outs[reshaped_avail_actions == 0] = -1e11

            # agent_outs = gumbel_softmax(agent_outs, hard=True)  # one-hot TODO ac_sample1
            # agent_outs = agent_outs.view(episode_num, self.n_agents, -1) # (1,3,9)
            # action_next = agent_outs.max(dim=2, keepdim=True)[1] # (1,3,1)
            # # action_next = torch.nonzero(agent_outs).squeeze()
            # actions_next_sample.append(action_next) # 选择动作的序号

            # agent_outs = gumbel_softmax(agent_outs, hard=True)  # 概率 TODO ac_mean
            agent_outs = F.softmax(agent_outs / 1, dim=1)  # 概率分布
            agent_outs = agent_outs.view(episode_num, self.n_agents, -1)
            actions_prob.append(agent_outs)  # 每个动作的概率

        actions_prob = torch.stack(actions_prob, dim=1)  # Concat over time
        # pi = actions_prob.view(-1, self.n_actions)
        # Calculated baseline
        baseline = torch.sum(actions_prob * q_targets, dim=-1).view(episode_num, max_episode_len,
                                                                    -1).detach()  # (1,60,3)
        q_targets_sample = baseline

        # actions_next_sample = torch.stack(actions_next_sample, dim=1)  # Concat over time # (1,60,3,1) TODO
        # # q_targets_sample = q_targets[:,:,actions_next]
        # q_targets_sample = torch.gather(q_targets, dim=3, index=actions_next_sample).squeeze(3) # (1,60,3)

        # # 得到target_q
        # q_targets[avail_u_next == 0.0] = - 9999999
        # q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_qmix_net(q_evals, s)  # [1, 60, 1]
        q_total_target = self.target_qmix_net(q_targets_sample, s_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        critic_loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if critic_loss.item() > 1e3:
            print("critic_loss:", critic_loss.item())

        # if train_step > 0 and train_step % self.args.target_update_cycle == 0: # 200
        #     self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        #     self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        #     self.target_policy.load_state_dict(self.agent.policy.state_dict()) #TODO
        # Update the frozen target models

        if train_step % self.args.save_cycle == 0:  # how often to save the model args.save_cycle = 5000
            self.save_model(train_step)  # TODO

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

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs,
                                                     self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

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
