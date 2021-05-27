import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    parser.add_argument('--alg', type=str, default='qmix',
                        help='the algorithm to train the agent')  # qmix reinforce+g2anet
    parser.add_argument('--last_action', type=bool, default=True,
                        help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
    parser.add_argument('--n_epoch', type=int, default=30000, help='n_epoch')
    parser.add_argument('--actor_buffer_size', type=int, default=32, help='actor_buffer_size')  # 32
    parser.add_argument('--critic_buffer_size', type=int, default=32, help='critic_buffer_size')  # 32
    parser.add_argument('--actor_update_delay', type=int, default=2, help='actor_update_delay')  # 2
    parser.add_argument('--actor_train_steps', type=int, default=1, help='actor_train_steps')  # 1
    parser.add_argument('--critic_lr', type=float, default=5e-4, help='critic_lr')  # 1
    parser.add_argument('--anneal_epsilon', type=float, default=2.4e-05,
                        help='critic_lr')  # 4.8e-05 10000 ;2.4e-05 20000;  1.6e-05 30000; 1.2e-05 40000
    parser.add_argument('--temp', type=float, default=1., help='softmax_temp')
    parser.add_argument('--loss_coeff_entropy', type=float, default=0.2, help='loss_coeff_entropy')  # 0.01
    parser.add_argument('--advantage_norm', type=bool, default=True, help='advantage_norm')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
    parser.add_argument('--auto_entropy', type=bool, default=True, help='auto_entropy')
    parser.add_argument('--buffer_size', type=int, default=5000, help='actor_buffer_size')  # 5000
    parser.add_argument('--ppo_buffer_size', type=int, default=32, help='actor_buffer_size')  # 5000
    parser.add_argument('--clip', type=float, default=0.2, help='')  # 5000
    parser.add_argument('--schedule_clip', type=str, default='linear', help='')  # 5000
    parser.add_argument('--actor_sample_times', type=int, default=5, help='actor_train_steps')  # 1
    parser.add_argument('--cuda_id', type=int, default=0)  # TODO
    args = parser.parse_args()
    return args


# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # the number of the epoch to train the agent
    args.n_epoch = 30000  # TODO

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4
    # args.critic_lr =  1e-4 #5e-4
    args.actor_lr = 5e-4

    # epsilon greedy
    # args.epsilon = 1
    # args.min_epsilon = 0.05
    # anneal_steps = 50000
    # args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    # args.epsilon_anneal_scale = 'step'

    # epsilon-greedy
    args.epsilon = 0.5
    # args.anneal_epsilon = 2.4e-05 # 4.8e-05 10000 ;2.4e-05 20000;  1.6e-05 30000; 1.2e-05 40000
    args.min_epsilon = 0.02  # lower-bounds the probability of any given action by episolin/|A|:
    args.epsilon_anneal_scale = 'epoch'  # 'episode'

    # the number of the epoch to train the agent
    # args.n_epoch = 30000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # the number of the train steps in one epoch
    args.train_steps = 1
    # args.actor_update_delay = 2
    # args.actor_train_steps = 1
    args.critic_train_steps = 1

    # # how often to evaluate
    args.evaluate_cycle = 100

    # experience replay
    args.critic_batch_size = 32  # 32
    args.actor_batch_size = 32  # on policy
    # args.critic_buffer_size = 32

    # args.critic_buffer_size = int(5e3)
    # args.actor_buffer_size = int(32) # on policy

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args
