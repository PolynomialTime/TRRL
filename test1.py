"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
#from imitation.data import rollout
import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

import gymnasium as gym

rng = np.random.default_rng(0)
env = make_vec_env(
    "CartPole-v1",
    n_envs=2,
    rng=rng,
    #post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)

x = env.unwrapped.envs[0].unwrapped.state
print(x)
if env.num_envs > 1:
    for idx in range(1, env.num_envs):
        x = np.append([x], [env.unwrapped.envs[idx].unwrapped.state], axis=0)
        
print(x)

 

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


def download_expert():
    print("Downloading a pretrained expert.")
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals-CartPole-v0",
        venv=env,
    )
    return expert


def sample_expert_transitions():
    expert = train_expert()  # uncomment to train your own expert
    #expert = download_expert()

    print("Sampling expert transitions.")

    
    rollouts = rollout.generate_trajectories(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=4),
        rng=rng,
        starting_state= None, #np.array([0.1, 0.1, 0.1, 0.1]),
        starting_action=None, #np.array([[1,], [1,],], dtype=np.integer)
    )
    
    '''
    rollouts = rollout.generate_transitions(
        expert,
        env,
        n_timesteps=5,
        rng=rng,
        starting_state=np.array([0.1, 0.1, 0.1, 0.1]),
        starting_action=np.array([0], dtype=np.integer),
        truncate=True,
    )
    '''
    return rollout.flatten_trajectories(rollouts)
    #return rollouts



transitions = sample_expert_transitions()

#transitions.obs.shape[0]
#x, y = transitions.obs.shape
print(len(transitions))
print(transitions)
'''
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)

evaluation_env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rng,
    env_make_kwargs={"render_mode": "human"},  # for rendering
)

print("Evaluating the untrained policy.")
reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
print(f"Reward before training: {reward}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=1)

print("Evaluating the trained policy.")
reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
print(f"Reward after training: {reward}")
'''