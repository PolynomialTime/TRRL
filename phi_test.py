"""This is the runner of using TRRL to infer the reward functions and the optimal policy

"""
import arguments
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import policies, MlpPolicy

from imitation.algorithms import bc
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

import gymnasium as gym

from reward_function import BasicRewardNet
import rollouts
from trrl import TRRL

from typing import (
    List,
)

arglist = arguments.parse_args()

rng = np.random.default_rng(0)
env = make_vec_env(
    arglist.env_name,
    n_envs=arglist.n_env,
    rng=rng,
    #post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)

print(arglist.env_name)
def train_expert():
    # note: use `download_expert` instead to download a pretrained, competent expert
    print("Training an expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.01,
        learning_rate=0.0001,
        gamma=0.99,
        n_epochs=20,
        n_steps=64,
    )
    expert.learn(100_000)  # Note: change this to 100_000 to train a decent expert.
    expert.save("expert_policy")
    return expert

def sample_expert_transitions(expert: policies):
    print("Sampling expert transitions.")
    trajs = rollouts.generate_trajectories(
        expert,
        env,
        rollouts.make_sample_until(min_timesteps=None, min_episodes=512),
        rng=rng,
        starting_state= None, #np.array([0.1, 0.1, 0.1, 0.1]),
        starting_action=None, #np.array([[1,], [1,],], dtype=np.integer)
    )

    return rollouts.flatten_trajectories(trajs)
    #return rollouts

#expert = train_expert()  # uncomment to train your own expert
expert = PPO.load("Lake_expert_policy")
#expert = PPO.load("Pole_expert_policy")

mean_reward, std_reward = evaluate_policy(model=expert, env=env)
print("Average reward of the expert is evaluated at: " + str(mean_reward) + ',' + str(std_reward) + '.')

transitions = sample_expert_transitions(expert)
print("Number of transitions in demonstrations: " + str(transitions.obs.shape[0]) + ".")
#print("transitions:",transitions)

obs = transitions.obs
actions = transitions.acts
infos = transitions.infos
next_obs = transitions.next_obs
dones = transitions.dones

rwd_net = BasicRewardNet(env.unwrapped.envs[0].unwrapped.observation_space, env.unwrapped.envs[0].unwrapped.action_space)

if arglist.device == 'cpu':
    DEVICE = torch.device('cpu')
elif arglist.device == 'gpu' and torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
elif arglist.device == 'gpu' and not torch.cuda.is_available():
    DEVICE = torch.device('cpu')
    print("Cuda is not available, run on CPU instead.")
else:
    DEVICE = torch.device('cpu')
    print("The intended device is not supported, run on CPU instead.")

# phi_test
# trrl_trainer = TRRL(
#     venv=env,
#     expert_policy=expert,
#     demonstrations=transitions,
#     demo_batch_size=arglist.demo_batch_size,
#     reward_net=rwd_net,
#     discount=arglist.discount,
#     avg_diff_coef=arglist.avg_reward_diff_coef,
#     l2_norm_coef=arglist.avg_reward_diff_coef,
#     l2_norm_upper_bound=arglist.l2_norm_upper_bound,
#     ent_coef=arglist.ent_coef,
#     device=DEVICE,
#     n_policy_updates_per_round=arglist.n_policy_updates_per_round,
#     n_reward_updates_per_round=arglist.n_reward_updates_per_round,
#     n_episodes_adv_fn_est=arglist.n_episodes_adv_fn_est,
#     n_timesteps_adv_fn_est=arglist.n_timesteps_adv_fn_est
# )
# print("Starting reward learning.")

# menglin
trrl_trainer = TRRL(
    venv=env,
    expert_policy=expert,
    demonstrations=transitions,
    demo_batch_size=arglist.demo_batch_size,
    reward_net=rwd_net,
    discount=arglist.discount,
    avg_diff_coef=arglist.avg_reward_diff_coef,
    l2_norm_coef=arglist.avg_reward_diff_coef,
    l2_norm_upper_bound=arglist.l2_norm_upper_bound,
    ent_coef=arglist.ent_coef,
    device=DEVICE,
    n_policy_updates_per_round=10,#arglist.n_policy_updates_per_round,
    n_reward_updates_per_round=3,#arglist.n_reward_updates_per_round,
    n_episodes_adv_fn_est=3,#arglist.n_episodes_adv_fn_est,
    n_timesteps_adv_fn_est=10#arglist.n_timesteps_adv_fn_est
)
print("Starting reward learning.")

trrl_trainer.train(n_rounds=arglist.n_runs)
