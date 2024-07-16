"""This is the runner of using TRRL to infer the reward functions and the optimal policy

"""
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.sac import policies as sac_policies
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.rewards.reward_nets import BasicRewardNet
from imitation.algorithms import bc
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

import gymnasium as gym

import rollouts
from trrl import TRRL

rng = np.random.default_rng(0)
env = make_vec_env(
    "CartPole-v1",
    n_envs=2,
    rng=rng,
    #post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)

def train_expert():
    # note: use `download_expert` instead to download a pretrained, competent expert
    print("Training a expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(1_000)  # Note: change this to 100_000 to train a decent expert.
    return expert

def sample_expert_transitions():
    expert = train_expert()  # uncomment to train your own expert
    #expert = download_expert()

    print("Sampling expert transitions.")

    trajs = rollouts.generate_trajectories(
        expert,
        env,
        rollouts.make_sample_until(min_timesteps=30, min_episodes=1),
        rng=rng,
        starting_state= None, #np.array([0.1, 0.1, 0.1, 0.1]),
        starting_action=None, #np.array([[1,], [1,],], dtype=np.integer)
    )
    
    return rollouts.flatten_trajectories(trajs)
    #return rollouts

transitions = sample_expert_transitions()

print(transitions.obs.shape)

rwd_net = BasicRewardNet(env.unwrapped.envs[0].unwrapped.observation_space, env.unwrapped.envs[0].unwrapped.action_space)

trrl_trainer = TRRL(
    venv=env,
    demonstrations=transitions,
    old_policy=sac_policies.SACPolicy,
    demo_batch_size=transitions.obs.shape[0],
    demo_minibatch_size=transitions.obs.shape[0],
    reward_net=rwd_net,
    discount=0.95,
    avg_diff_coef=0.1,
    l2_norm_coef=0.1,
    l2_norm_upper_bound=0.05,
    ent_coef=0.1
)

trrl_trainer.train(n_rounds=2)