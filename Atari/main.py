"""This is the runner of using TRRL to infer the reward functions and the optimal policy

"""
import multiprocessing as mp
import arguments
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import policies, MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage

from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

import gymnasium as gym

from reward_function import BasicRewardNet,CnnRewardNet
from trrl import TRRL
from imitation.data import rollout

from typing import (
    List,
)


def train_expert():
    # note: use `download_expert` instead to download a pretrained, competent expert
    print("Training an expert.")
    expert = PPO(
        policy="CnnPolicy",
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=arglist.ent_coef,
        learning_rate=arglist.lr,
        gamma=arglist.discount,
        n_epochs=20,
        n_steps=128
    )
    expert.learn(100_00000)  # Note: change this to 100_000 to train a decent expert.
    expert.save(f"./expert_data/{arglist.env_name}")
    return expert


def sample_expert_transitions(expert: policies):
    print("Sampling expert transitions.")
    trajs = rollout.generate_trajectories(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=arglist.transition_truncate_len),
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(trajs)

    torch.save(transitions, f"./expert_data/transitions_{arglist.env_name}.npy")
    # torch.save(rollouts,f"./imitation/imitation_expert/rollouts_{env_name}.npy")

    return transitions


if __name__ == '__main__':

    # make environment
    #mp.set_start_method('spawn', force=True)
    arglist = arguments.parse_args()

    rng = np.random.default_rng(0)

    env = make_atari_env(
        arglist.env_name,      # 环境名称，例如 "PongNoFrameskip-v4"
        n_envs=arglist.n_env,  # 并行环境的数量
    )

    # 添加帧堆叠 (默认堆叠最近 4 帧)
    env = VecFrameStack(env, n_stack=4)

    # 添加通道转置 (将观察值从 HWC 转为 CHW 格式)
    env = VecTransposeImage(env)
    print(arglist.env_name)
    print(env.observation_space.shape)


    print(type(env))

    # load expert data

    expert = PPO.load(f"./expert_data/{arglist.env_name}")
    transitions = torch.load(f"./expert_data/transitions_{arglist.env_name}.npy")

    # TODO: If the environment is running for the first time (i.e., no expert data is present in the folder), please execute the following code first.
    #expert = train_expert()  # uncomment to train your own expert
   

    mean_reward, std_reward = evaluate_policy(model=expert, env=env)
    print("Average reward of the expert is evaluated at: " + str(mean_reward) + ',' + str(std_reward) + '.')
    

    #transitions = sample_expert_transitions(expert)
    print("Number of transitions in demonstrations: " + str(transitions.obs.shape[0]) + ".")

    # @truncate the length of expert transition
    transitions = transitions[:arglist.transition_truncate_len]
    #print(transitions)

    obs = transitions.obs
    actions = transitions.acts
    infos = transitions.infos
    next_obs = transitions.next_obs
    dones = transitions.dones
    print(f"obs shape: {transitions.obs.shape}")
    print(f"actions shape: {transitions.acts.shape}")
    #print(f"infos type and structure: {type(transitions.infos)}")  # 因为 infos 通常是字典
    print(f"next_obs shape: {transitions.next_obs.shape}")
    print(f"dones shape: {transitions.dones.shape}")


    # initiate reward_net
    # env_spec = gym.spec(arglist.env_name)
    # env_temp = env_spec.make()
    # observation_space = env_temp.observation_space
    # action_space = env_temp.action_space
    #rwd_net = CnnRewardNet(observation_space, action_space)
    # 从实际环境中提取 observation_space 和 action_space
    observation_space = env.observation_space
    action_space = env.action_space

    # 初始化 CnnRewardNet
    rwd_net = BasicRewardNet(observation_space, action_space)
    #print("observation_space", observation_space, ",action_space", action_space)
    # 打印提取的空间信息，供调试使用
    print("Actual observation_space:", observation_space)
    print("Actual action_space:", action_space)
    
    # initiate device
    if arglist.device == 'cpu':
        DEVICE = torch.device('cpu')
    elif arglist.device == 'gpu' and torch.cuda.is_available():
        DEVICE = torch.device('cuda:1')
    elif arglist.device == 'gpu' and not torch.cuda.is_available():
        DEVICE = torch.device('cpu')
        print("Cuda is not available, run on CPU instead.")
    else:
        DEVICE = torch.device('cpu')
        print("The intended device is not supported, run on CPU instead.")

    # mian func
    trrl_trainer = TRRL(
        venv=env,
        expert_policy=expert,
        demonstrations=transitions,
        demo_batch_size=arglist.demo_batch_size,
        reward_net=rwd_net,
        discount=arglist.discount,
        avg_diff_coef=arglist.avg_reward_diff_coef,
        l2_norm_coef=arglist.l2_norm_coef,
        l2_norm_upper_bound=arglist.l2_norm_upper_bound,
        target_reward_diff=arglist.target_reward_diff,
        target_reward_l2_norm=arglist.target_reward_l2_norm,
        ent_coef=arglist.ent_coef,
        device=DEVICE,
        n_policy_updates_per_round=arglist.n_policy_updates_per_round,
        n_reward_updates_per_round=arglist.n_reward_updates_per_round,
        n_episodes_adv_fn_est=arglist.n_episodes_adv_fn_est,
        n_timesteps_adv_fn_est=arglist.n_timesteps_adv_fn_est,
        observation_space=observation_space,
        action_space=action_space,
        arglist=arglist
    )
    print("Starting reward learning.")

    trrl_trainer.train(n_rounds=arglist.n_global_rounds)
