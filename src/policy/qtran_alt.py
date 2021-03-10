import torch
import torch.nn as nn
import os
from network.base_net import RNN
from network.qtran_net import QtranV, QtranQAlt


class QtranAlt:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        rnn_input_shape = self.obs_shape

        # 根据参数决定RNN的输入维度
        if args.last_action:
            rnn_input_shape += self.n_actions  # 当前agent的上一个动作的one_hot向量
        if args.reuse_network:
            rnn_input_shape += self.n_agents
        self.args = args
        # 神经网络
        self.eval_rnn = RNN(rnn_input_shape, args)  # individual networks
        self.target_rnn = RNN(rnn_input_shape, args)

        self.eval_joint_q = QtranQAlt(args)  # counterfactual joint networks
        self.target_joint_q = QtranQAlt(args)
        self.v = QtranV(args)

        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_joint_q.cuda()
            self.target_joint_q.cuda()
            self.v.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_joint_q = self.model_dir + '/joint_q_params.pkl'
                path_v = self.model_dir + '/v_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_joint_q.load_state_dict(torch.load(path_joint_q, map_location=map_location))
                self.v.load_state_dict(torch.load(path_v, map_location=map_location))
                print('Successfully load the model: {}, {} and {}'.format(path_rnn, path_joint_q, path_v))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_joint_q.load_state_dict(self.eval_joint_q.state_dict())

        self.eval_parameters = list(self.eval_joint_q.parameters()) + \
                               list(self.v.parameters()) + \
                               list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg QTRAN-alt')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
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
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']
        mask = 1 - batch["padded"].float().repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        if self.args.cuda:
            u = u.cuda()
            r = r.cuda()
            avail_u = avail_u.cuda()
            avail_u_next = avail_u_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # 得到每个agent对应的Q和hidden_states，维度为(episode个数, max_episode_len， n_agents， n_actions/hidden_dim)
        individual_q_evals, individual_q_targets, hidden_evals, hidden_targets = self._get_individual_q(batch, max_episode_len)

        # 得到当前时刻和下一时刻每个agent的局部最优动作及其one_hot表示
        individual_q_clone = individual_q_evals.clone()
        individual_q_clone[avail_u == 0.0] = - 999999
        individual_q_targets[avail_u_next == 0.0] = - 999999

        opt_onehot_eval = torch.zeros(*individual_q_clone.shape)
        opt_action_eval = individual_q_clone.argmax(dim=3, keepdim=True)
        opt_onehot_eval = opt_onehot_eval.scatter(-1, opt_action_eval[:, :].cpu(), 1)

        opt_onehot_target = torch.zeros(*individual_q_targets.shape)
        opt_action_target = individual_q_targets.argmax(dim=3, keepdim=True)
        opt_onehot_target = opt_onehot_target.scatter(-1, opt_action_target[:, :].cpu(), 1)

        # ---------------------------------------------L_td-------------------------------------------------------------

        # 计算joint_q和v，要注意joint_q是每个agent都有，v只有一个
        # joint_q的维度为(episode个数, max_episode_len， n_agents， n_actions), 而且joint_q在后面的l_nopt还要用到
        # v的维度为(episode个数, max_episode_len)
        joint_q_evals, joint_q_targets, v = self.get_qtran(batch, opt_onehot_target, hidden_evals, hidden_targets)

        # 取出当前agent动作对应的joint_q_chosen以及它的局部最优动作对应的joint_q
        joint_q_chosen = torch.gather(joint_q_evals, dim=-1, index=u).squeeze(-1)  # (episode个数, max_episode_len, n_agents)
        joint_q_opt = torch.gather(joint_q_targets, dim=-1, index=opt_action_target).squeeze(-1)

        # loss
        y_dqn = r.repeat(1, 1, self.n_agents) + self.args.gamma * joint_q_opt * (1 - terminated.repeat(1, 1, self.n_agents))
        td_error = joint_q_chosen - y_dqn.detach()
        l_td = ((td_error * mask) ** 2).sum() / mask.sum()
        # ---------------------------------------------L_td-------------------------------------------------------------

        # ---------------------------------------------L_opt------------------------------------------------------------

        # 将局部最优动作的Q值相加  (episode个数,max_episode_len)
        # 这里要使用individual_q_clone，它把不能执行的动作Q值改变了，使用individual_q_evals可能会使用不能执行的动作的Q值
        q_sum_opt = individual_q_clone.max(dim=-1)[0].sum(dim=-1)

        # 重新得到joint_q_opt_eval，它和joint_q_evals的区别是前者输入的动作是当前局部最优动作，后者输入的动作是当前执行的动作
        joint_q_opt_evals, _, _ = self.get_qtran(batch, opt_onehot_eval, hidden_evals, hidden_targets, hat=True)
        joint_q_opt_evals = torch.gather(joint_q_opt_evals, dim=-1, index=opt_action_eval).squeeze(-1)  # (episode个数, max_episode_len， n_agents)

        # 因为QTRAN-alt要对每个agent都计算l_opt，所以要把q_sum_opt和v再增加一个agent维
        q_sum_opt = q_sum_opt.unsqueeze(-1).expand(-1, -1, self.n_agents)
        v = v.unsqueeze(-1).expand(-1, -1, self.n_agents)
        opt_error = q_sum_opt - joint_q_opt_evals.detach() + v  # 计算l_opt时需要将joint_q_opt_evals固定
        l_opt = ((opt_error * mask) ** 2).sum() / mask.sum()

        # ---------------------------------------------L_opt------------------------------------------------------------

        # ---------------------------------------------L_nopt-----------------------------------------------------------
        # 因为L_nopt约束的是当前agent所有可执行的动作中，对应的最小的d，为了让不能执行的动作不影响d的计算，将不能执行的动作对应的q变大
        individual_q_evals[avail_u == 0.0] = 999999

        # 得到agent_i之外的其他agent的执行动作的Q值之和q_other_sum
        #   1. 先得到每个agent的执行动作的Q值q_all,(episode个数, max_episode_len, n_agents， 1)
        q_all_chosen = torch.gather(individual_q_evals, dim=-1, index=u)
        #   2. 把q_all最后一个维度上当前agent的Q值变成所有agent的Q值，(episode个数, max_episode_len, n_agents, n_agents)
        q_all_chosen = q_all_chosen.view((episode_num, max_episode_len, 1, -1)).repeat(1, 1, self.n_agents, 1)
        q_mask = (1 - torch.eye(self.n_agents)).unsqueeze(0).unsqueeze(0)
        if self.args.cuda:
            q_mask = q_mask.cuda()
        q_other_chosen = q_all_chosen * q_mask  # 把每个agent自己的Q值置为0，从而才能相加得到其他agent的Q值之和
        #   3. 求和，同时由于对于当前agent的每个动作，都要和q_other_sum相加，所以把q_other_sum扩展出n_actions维度
        q_other_sum = q_other_chosen.sum(dim=-1, keepdim=True).repeat(1, 1, 1, self.n_actions)

        # 当前agent的每个动作的Q和其他agent执行动作的Q相加，得到D中的第一项
        q_sum_nopt = individual_q_evals + q_other_sum

        # 因为joint_q_evals的维度是(episode个数,max_episode_len,n_agents,n_actions)，所以要对v扩展出一个n_actions维度
        v = v.unsqueeze(-1).expand(-1, -1, -1, self.n_actions)
        d = q_sum_nopt - joint_q_evals.detach() + v  # 计算l_nopt时需要将qtran_q_evals固定
        d = d.min(dim=-1)[0]
        l_nopt = ((d * mask) ** 2).sum() / mask.sum()
        # ---------------------------------------------L_nopt-----------------------------------------------------------

        # print('l_td is {}, l_opt is {}, l_nopt is {}'.format(l_td, l_opt, l_nopt))
        loss = l_td + self.args.lambda_opt * l_opt + self.args.lambda_nopt * l_nopt
        # loss = l_td + self.args.lambda_opt * l_opt
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_joint_q.load_state_dict(self.eval_joint_q.state_dict())

    def _get_individual_q(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets, hidden_evals, hidden_targets = [], [], [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_individual_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                inputs_next = inputs_next.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
            hidden_eval, hidden_target = self.eval_hidden.clone(), self.target_hidden.clone()

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            hidden_eval = hidden_eval.view(episode_num, self.n_agents, -1)
            hidden_target = hidden_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
            hidden_evals.append(hidden_eval)
            hidden_targets.append(hidden_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        hidden_evals = torch.stack(hidden_evals, dim=1)
        hidden_targets = torch.stack(hidden_targets, dim=1)
        return q_evals, q_targets, hidden_evals, hidden_targets

    def _get_individual_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号
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
        return inputs, inputs_next

    def get_qtran(self, batch, local_opt_actions, hidden_evals, hidden_targets=None, hat=False):
        episode_num, max_episode_len, _, _ = hidden_evals.shape
        s = batch['s'][:, :max_episode_len]
        s_next = batch['s_next'][:, :max_episode_len]
        u_onehot = batch['u_onehot'][:, :max_episode_len]
        v_state = s.clone()

        # s和s_next没有n_agents维度，每个agent的joint_q网络都需要, 所以要把s转化成四维
        s = s.unsqueeze(-2).expand(-1, -1, self.n_agents, -1)
        s_next = s_next.unsqueeze(-2).expand(-1, -1, self.n_agents, -1)
        # 添加agent编号对应的one-hot向量
        '''
        因为当前的inputs三维的数据，每一维分别代表(episode编号，agent编号，inputs维度)，直接在后面添加对应的向量
        即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
        agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
        '''
        action_onehot = torch.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand(episode_num, max_episode_len, -1, -1)
        s_eval = torch.cat([s, action_onehot], dim=-1)
        s_target = torch.cat([s_next, action_onehot], dim=-1)
        if self.args.cuda:
            s_eval = s_eval.cuda()
            s_target = s_target.cuda()
            v_state = v_state.cuda()
            u_onehot = u_onehot.cuda()
            hidden_evals = hidden_evals.cuda()
            hidden_targets = hidden_targets.cuda()
            local_opt_actions = local_opt_actions.cuda()
        if hat:
            # 神经网络输出的q_eval、q_target的维度为(episode_num * max_episode_len * n_agents, n_actions)
            q_evals = self.eval_joint_q(s_eval, hidden_evals, local_opt_actions)
            q_targets = None
            v = None

            # 把q_eval维度变回(episode_num, max_episode_len, n_agents, n_actions)
            q_evals = q_evals.view(episode_num, max_episode_len, -1, self.n_actions)
        else:
            q_evals = self.eval_joint_q(s_eval, hidden_evals, u_onehot)
            q_targets = self.target_joint_q(s_target, hidden_targets, local_opt_actions)
            v = self.v(v_state, hidden_evals)
            # 把q_eval、q_target维度变回(episode_num, max_episode_len, n_agents, n_actions)
            q_evals = q_evals.view(episode_num, max_episode_len, -1, self.n_actions)
            q_targets = q_targets.view(episode_num, max_episode_len, -1, self.n_actions)
            # 把v维度变回(episode_num, max_episode_len)
            v = v.view(episode_num, -1)

        return q_evals, q_targets, v

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
        torch.save(self.eval_joint_q.state_dict(), self.model_dir + '/' + num + '_joint_q_params.pkl')
        torch.save(self.v.state_dict(), self.model_dir + '/' + num + '_v_params.pkl')
