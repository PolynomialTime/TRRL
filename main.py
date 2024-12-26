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

from imitation.algorithms import bc
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

import gymnasium as gym

from reward_function import BasicRewardNet
from trrl import TRRL
from imitation.data import rollout

from typing import (
    List,
)


def train_expert():
    # note: use `download_expert` instead to download a pretrained, competent expert
    print("Training an expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=arglist.ent_coef,
        learning_rate=arglist.lr,
        gamma=arglist.discount,
        n_epochs=20,
        n_steps=128
    )
    expert.learn(100_000)  # Note: change this to 100_000 to train a decent expert.
    expert.save(f"./expert_data/{arglist.env_name}")
    return expert


def sample_expert_transitions(expert: policies):
    print("Sampling expert transitions.")
    trajs = rollout.generate_trajectories(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=512),
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(trajs)

    torch.save(transitions, f"./expert_data/transitions_{arglist.env_name}.npy")
    # torch.save(rollouts,f"./imitation/imitation_expert/rollouts_{env_name}.npy")

    return transitions


if __name__ == '__main__':

    # make environment
    mp.set_start_method('spawn', force=True)
    arglist = arguments.parse_args()

    rng = np.random.default_rng(0)

    env = make_vec_env(
        arglist.env_name,
        n_envs=arglist.n_env,
        rng=rng,
        parallel=True,
        max_episode_steps=500,
    )

    print(arglist.env_name)

    print(type(env))

    # load expert data

    expert = PPO.load(f"./expert_data/{arglist.env_name}")
    transitions = torch.load(f"./expert_data/transitions_{arglist.env_name}.npy")

    # TODO: If the environment is running for the first time (i.e., no expert data is present in the folder), please execute the following code first.
    # expert = train_expert()  # uncomment to train your own expert
    # transitions = sample_expert_transitions(expert)

    mean_reward, std_reward = evaluate_policy(model=expert, env=env)
    print("Average reward of the expert is evaluated at: " + str(mean_reward) + ',' + str(std_reward) + '.')
    print("Number of transitions in demonstrations: " + str(transitions.obs.shape[0]) + ".")

    # @truncate the length of expert transition
    transitions = transitions[:arglist.transition_truncate_len]
    #print(transitions)

    obs = transitions.obs
    actions = transitions.acts
    infos = transitions.infos
    next_obs = transitions.next_obs
    dones = transitions.dones

    # initiate reward_net
    env_spec = gym.spec(arglist.env_name)
    env_temp = env_spec.make()
    observation_space = env_temp.observation_space
    action_space = env_temp.action_space
    rwd_net = BasicRewardNet(observation_space, action_space)
    print("observation_space", observation_space, ",action_space", action_space)

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
