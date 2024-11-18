"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
# test
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

import gymnasium as gym

from imitation.rewards.reward_nets import BasicRewardNet
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from reward_function import RwdFromRwdNet

'''
rng = np.random.default_rng(0)
env = make_vec_env(
    "CartPole-v1",
    n_envs=4,
    rng=rng,
    #post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)

_ = env.reset()


rwd_net = BasicRewardNet(env.unwrapped.envs[0].unwrapped.observation_space, env.unwrapped.envs[0].unwrapped.action_space)
rwd_fn = RwdFromRwdNet(rwd_net=rwd_net)

wenv = RewardVecEnvWrapper(
    venv=env,
    reward_fn=rwd_fn
)

starting_action = np.array([0], dtype=np.integer)

#actions = np.repeat(starting_action, repeats=[4], axis=0)

#print(wenv.step(actions))

loss = torch.zeros((1,1))
print(loss)
'''
for i in range(0, 10, 3):
    print(i)
