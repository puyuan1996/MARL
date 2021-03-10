import torch
import torch.nn as nn
import torch.nn.functional as f


# output prob of z for an episode
class HierarchicalPolicy(nn.Module):
    def __init__(self, args):
        super(HierarchicalPolicy, self).__init__()
        self.fc_1 = nn.Linear(args.state_shape, 128)
        self.fc_2 = nn.Linear(128, args.noise_dim)

    def forward(self, state):
        x = f.relu(self.fc_1(state))
        q = self.fc_2(x)
        prob = f.softmax(q, dim=-1)
        return prob


class BootstrappedRNN(nn.Module):
    def __init__(self, input_shape, args):
        super(BootstrappedRNN, self).__init__()
        self.args = args

        self.fc = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.hyper_w = nn.Linear(args.noise_dim + args.n_agents, args.rnn_hidden_dim * args.n_actions)
        self.hyper_b = nn.Linear(args.noise_dim + args.n_agents, args.n_actions)

    def forward(self, obs, hidden_state, z):
        agent_id = obs[:, -self.args.n_agents:]
        hyper_input = torch.cat([z, agent_id], dim=-1)

        x = f.relu(self.fc(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        h = h.view(-1, 1, self.args.rnn_hidden_dim)

        hyper_w = self.hyper_w(hyper_input)
        hyper_b = self.hyper_b(hyper_input)
        hyper_w = hyper_w.view(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        hyper_b = hyper_b.view(-1, 1, self.args.n_actions)

        q = torch.bmm(h, hyper_w) + hyper_b
        q = q.view(-1, self.args.n_actions)
        return q, h


# variational distribution for MI Loss， output q(z|sigma(tau))
class VarDistribution(nn.Module):
    def __init__(self, args):
        super(VarDistribution, self).__init__()
        self.args = args

        self.GRU = nn.GRU(args.n_agents * args.n_actions + args.state_shape, 64)

        self.fc_1 = nn.Linear(64, 32)
        self.fc_2 = nn.Linear(32, args.noise_dim)

    def forward(self, q_value, avail_actions, state, episode_length):  # q_value.
        """
        :param q_value:   shape = (episode_num, episode_limit, n_agents, n_actions)
        :param avail_actions:
        :param state:     shape = (episode_num, episode_limit, state_shape)
        :param episode_final_idx:     shape = (episode_num)
        :return:          q(z|sigma(tau))
        """
        episode_num = q_value.shape[0]
        output = []
        # get sigma(q) by softmax
        # the length of each episode are different，so we can't do it in one forward
        for i in range(episode_num):
            length = episode_length[i]
            # only take the valid data
            q,  avail_action, s = q_value[i, :length], avail_actions[i, :length], state[i, :length]
            q = f.softmax(q, dim=-1)
            q = q * avail_action
            q = q / q.sum(dim=-1, keepdim=True)

            q = q.reshape(1, length, -1)
            s = s.unsqueeze(0)
            inputs = torch.cat([q, s], dim=-1)
            inputs = inputs.permute(1, 0, 2)

            _, h = self.GRU(inputs)  # (1, 1, 64)
            x = f.relu(self.fc_1(h.squeeze(0)))
            x = self.fc_2(x)
            x = f.softmax(x, dim=-1)
            output.append(x)
        output = torch.cat(output, dim=0)
        return output
