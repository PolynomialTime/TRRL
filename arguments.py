import time
import argparse

time_now = time.strftime('%Y_%m_%d_%H:%M')


# python3 main.py --device gpu --n_episodes_adv_fn_est 32 --n_timesteps_adv_fn_est 32 --env_name FrozenLake-v1 --n_env 8

def parse_args():
    parser = argparse.ArgumentParser("Proximal Inverse Reward Optimization")

    # system info
    parser.add_argument("--time", type=str, default=time_now, help="system time")
    parser.add_argument("--device", type=str, default='cpu', help="torch device")

    # environment
    # Ant-v4, HalfCheetah-v4, Hopper-v3, Walker2d-v3, Pendulum-v1, Acrobot-v1, BipedalWalker-v3, FrozenLake-v1, CartPole-v1，MountainCar-v0
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="the environment")
    parser.add_argument("--n_env", type=int, default=1, help="number of parallel envs in venvs")
    parser.add_argument("--n_episodes", type=int, default=8, help="number of episodes for sample trajectory")
    parser.add_argument("--discount", type=float, default=0.99, help="discount factor")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument("--demo_batch_size", type=int, default=64, help="number of demos to generate")
    parser.add_argument("--policy_model", type=str, default='SAC', help="PPO or SAC for the policy model")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    
    # core training parameters
    parser.add_argument("--max_epoch", type=int, default=100, help="maximum epoch length")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for adam optimizer")
    parser.add_argument("--avg_reward_diff_coef", type=float, default=0.1,
                        help="Langrange multiplier for the average difference between the old and new reward function")
    parser.add_argument("--l2_norm_coef", type=float, default=0.1,
                        help="Langrange multiplier for the l2 norm of the difference between the old and new reward function")
    parser.add_argument("--l2_norm_upper_bound", type=float, default=0.1,
                        help="upper bound for the l2 norm of the difference between the old and new reward function")

    parser.add_argument("--ppo_n_epochs", type=int, default=5, help="number of epochs for PPO updates")
    parser.add_argument("--buffer_size", type=int, default=400, help="maximum buffer size per key for trajectory storage")
    parser.add_argument("--coef_scale_up", type=float, default=1.2, help="scale up factor for coefficient adjustment")
    parser.add_argument("--coef_scale_down", type=float, default=1.2, help="scale down factor for coefficient adjustment")
    parser.add_argument("--l2_coef_scale_up", type=float, default=2, help="scale up factor for l2 coefficient adjustment")
    parser.add_argument("--l2_coef_scale_down", type=float, default=2, help="scale down factor for l2 coefficient adjustment (1/1.2)")
    parser.add_argument("--target_ratio_upper", type=float, default=1.5, help="upper ratio for target reward difference threshold")
    parser.add_argument("--target_ratio_lower", type=float, default=1.5, help="lower ratio for target reward difference threshold (1/1.5)")
    parser.add_argument("--coef_min", type=float, default=1e-3, help="minimum value for coefficient")
    parser.add_argument("--coef_max", type=float, default=1e2, help="maximum value for coefficient")

    # adaptive coef adjustment paremeters
    parser.add_argument("--target_reward_diff", type=float, default=0.005,
                        help="threshold for dynamic adjustment of Lagrange multiplier for soft Bellan error.")
    parser.add_argument("--target_reward_l2_norm", type=float, default=0.1,
                        help="threshold for dynamic adjustment of Lagrange multiplier for l2 norm of reward difference.")


    # experiment control parameters
    parser.add_argument("--n_global_rounds", type=int, default=1000, help="number of global rounds")
    parser.add_argument("--n_policy_updates_per_round", type=int, default=100000,
                        help="number of policy udpates per global round")
    parser.add_argument("--n_reward_updates_per_round", type=int, default=1,
                        help="number of reward udpates per global round")

    # checkpointing
    parser.add_argument("--fre4save_model", type=int, default=100,
                        help="the number of the episode for saving the model")
    parser.add_argument("--start_save_model", type=int, default=100,
                        help="the number of the epoches for saving the model")
    parser.add_argument("--start_save_log", type=int, default=1,
                        help="interval of epoch for saving the log")
    parser.add_argument("--save_model_dir", type=str, default="./model_saved/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--load_model_dir", type=str, default="./model_saved/",
                        help="directory in which training state and model are loaded"),
    parser.add_argument("--save_results_dir", type=str, default="./output/",
                        help="directory which results are output to")

    parser.add_argument("--transition_truncate_len", type=int, default=1024,
                        help="truncate transitions")

    parser.add_argument("--mc_interval", type=int, default=10,
                        help="Monte Carlo sample interval")
    return parser.parse_args()
