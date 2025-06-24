"""This is the runner of using PIRO to infer the reward functions and the optimal policy

"""
import multiprocessing as mp
import arguments
import torch
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import policies, MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize

from imitation.algorithms import bc
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

import gymnasium as gym

from reward_function import BasicRewardNet
from piro import PIRO
from imitation.data import rollout

from typing import (
    List,
)


def train_expert():
    # note: use `download_expert` instead to download a pretrained, competent expert
    print("Training an expert.")
    
    if arglist.policy_model.upper() == 'PPO':
        expert = PPO(
            policy="MlpPolicy",
            env=env,
            seed=0,
            batch_size=64,
            ent_coef=arglist.ent_coef,
            learning_rate=arglist.lr,
            gamma=arglist.discount,
            n_epochs=20,
            n_steps=128
        )
    elif arglist.policy_model.upper() == 'SAC':
        expert = SAC(
            policy="MlpPolicy",
            batch_size=256,
            env=env,
            learning_rate=3e-4,
            ent_coef="auto",
            verbose=0,
            device='cpu',
        )
    else:
        raise ValueError(f"Unsupported policy model: {arglist.policy_model}")

    expert.learn(1000_000, progress_bar=True)  # Note: change this to 100_000 to train a decent expert.
    expert.save(f"./expert_data/{arglist.env_name}")
    return expert


def sample_expert_transitions(expert: policies):
    print("Sampling expert transitions.")
    trajs = rollout.generate_trajectories(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=1),
        rng=rng,
    )

    total_reward = sum([sum(traj.rews) for traj in trajs])
    print(f"Number of trajectories: {len(trajs)}")
    print(f"Number of transitions: {sum([len(traj) for traj in trajs])}")
    print(f"Average reward per trajectory: {total_reward / len(trajs)}")

    transitions = rollout.flatten_trajectories(trajs)

    torch.save(transitions, f"./expert_data/transitions_{arglist.env_name}.npy")
    # torch.save(rollouts,f"./imitation/imitation_expert/rollouts_{env_name}.npy")

    return transitions


if __name__ == '__main__':    # make environment
    mp.set_start_method('spawn', force=True)
    arglist = arguments.parse_args()
    
    arglist.save_results_dir = f"./output/{arglist.env_name}/{arglist.transition_truncate_len}/{arglist.seed}/"

    rng = np.random.default_rng(arglist.seed)

    env = make_vec_env(
        arglist.env_name,
        n_envs=arglist.n_env,
        rng=rng,
        #parallel=True,
        #max_episode_steps=500,
    )
    #env = VecNormalize(env, norm_obs=True, norm_reward=False)
    print(arglist.env_name)
    print(type(env))

    # TODO: If the environment is running for the first time (i.e., no expert data is present in the folder), please execute the following code first.
    # Load expert data or train new expert if not exists
    expert_path = f"./expert_data/{arglist.env_name}.zip"
    transitions_path = f"./expert_data/transitions_{arglist.env_name}.npy"
    
    try:
        # Try to load existing expert model based on policy_model parameter
        if arglist.policy_model.upper() == 'PPO':
            expert = PPO.load(expert_path)
            print(f"Loaded existing PPO expert from {expert_path}")
        elif arglist.policy_model.upper() == 'SAC':
            expert = SAC.load(expert_path)
            print(f"Loaded existing SAC expert from {expert_path}")
        else:
            raise ValueError(f"Unsupported policy model: {arglist.policy_model}")
    except (FileNotFoundError, Exception) as e:
        print(f"Expert model not found or failed to load: {e}")
        print("Training new expert...")
        expert = train_expert()
    
    try:
        # Try to load existing transitions
        transitions = torch.load(transitions_path)
        print(f"Loaded existing transitions from {transitions_path}")
    except (FileNotFoundError, Exception) as e:
        print(f"Transitions not found or failed to load: {e}")
        print("Sampling new expert transitions...")
        transitions = sample_expert_transitions(expert)

    mean_reward, std_reward = evaluate_policy(model=expert, env=env)
    print("Average reward of the expert is evaluated at: " + str(mean_reward) + ',' + str(std_reward) + '.')
    print("Number of transitions in demonstrations: " + str(transitions.obs.shape[0]) + ".")

    # @truncate the length of expert transition
    if arglist.transition_truncate_len is not None and arglist.transition_truncate_len > 0:
        transitions = transitions[:arglist.transition_truncate_len]
    else:       
        print("No truncation of transitions is applied.")
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
    trrl_trainer = PIRO(
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
        observation_space=observation_space,
        action_space=action_space,
        arglist=arglist
    )
    print("Starting reward learning.")

    trrl_trainer.train(n_rounds=arglist.n_global_rounds)
