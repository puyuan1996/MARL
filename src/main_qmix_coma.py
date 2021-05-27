from runner_qmix import Runner
# from runner_coma import Runner # to run mCOMA method
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, \
    get_commnet_args, get_g2anet_args
import torch
import numpy as np

if __name__ == '__main__':
    for i in range(5):
        args = get_common_args()
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)

        # Set seeds
        args.seed = i
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            seed=args.seed,
                            replay_dir=args.replay_dir)

        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        args.epsilon_start = 1  # 0.5
        args.epsilon_finish = 0.05
        args.epsilon_anneal_time = 50000

        print(f'map:{args.map},episode_limit:{args.episode_limit}', print(f'cuda_id:{args.cuda_id}'))
        print(f'n_epoch:{args.n_epoch}')

        torch.cuda.set_device(args.cuda_id)
        runner = Runner(env, args)
        if args.learn:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
