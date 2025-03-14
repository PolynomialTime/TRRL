"""This is the runner of using TRRL to infer the reward functions and the optimal policy

"""
import pandas as pd
import tqdm
import datetime
import arguments
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import policies, MlpPolicy
from stable_baselines3.common import (
    base_class,
    distributions,
    on_policy_algorithm,
    policies,
    vec_env,
    evaluation
)
import copy
from imitation.util.util import make_vec_env
from imitation.algorithms.adversarial.common import AdversarialTrainer
from imitation.algorithms.adversarial.gail import GAIL
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

import gymnasium as gym
from reward_function import RwdFromRwdNet, RewardNet
from reward_function import BasicRewardNet
import rollouts
import os
import torch.utils.tensorboard as tb
from imitation.algorithms.sqil import SQIL
# from sqil import SQIL
from trrl import TRRL
#from BC import BC
from typing import (
    List,
)
import torch.nn.functional as F
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util import util
import logging
from imitation.policies.serialize import load_policy
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack
from imitation.rewards.reward_nets import BasicShapedRewardNet
from stable_baselines3.common.env_util import make_atari_env

logging.basicConfig(level=logging.WARNING)
class NullLogger(logging.Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle(self, record):
        # 忽略所有日志记录
        pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建空日志记录器实例
null_logger = NullLogger(name="null")
rng = np.random.default_rng(0)

arglist = arguments.parse_args()

env = make_atari_env(
        arglist.env_name,      # 环境名称，例如 "PongNoFrameskip-v4"
        n_envs=arglist.n_env,  # 并行环境的数量
    )
# 添加帧堆叠 (默认堆叠最近 4 帧)
env = VecFrameStack(env, n_stack=4)
print(f"Environment set to: {arglist.env_name}")
env = VecTransposeImage(env)
print(arglist.env_name)

# Sample transitions from the expert policy
expert = PPO.load(f"./expert_data/{arglist.env_name}")
# Evaluate expert policy
mean_reward, std_reward = evaluate_policy(model=expert, env=env)
print(f"Average reward of the expert: {mean_reward}, {std_reward}.")

transitions = torch.load(f"./expert_data/transitions_{arglist.env_name}.npy")
print(f"Number of transitions in demonstrations: {transitions.obs.shape[0]}.")

transitions = transitions[:arglist.transition_truncate_len]
print(f"Number of transitions after: {transitions.obs.shape[0]}.")

# 打印生成器环境的观察值形状
sample_obs = env.reset()
print(f"Generated observation shape (after reset): {sample_obs.shape}")

rwd_net = BasicShapedRewardNet(
    env.observation_space,
    env.action_space,
)

# Initialize TensorBoard logger
writer = tb.SummaryWriter(log_dir=f"logs/{arglist.env_name}/SQIL", flush_secs=1)

def _evaluate_policy(sqil_trainer, env) -> float:
    """Evalute the true expected return of the learned policy under the original environment.

    :return: The true expected return of the learning policy.
    """

    mean_reward, std_reward = evaluation.evaluate_policy(model=sqil_trainer.policy, env=env)


    return mean_reward


def expert_kl(sqil_trainer, expert, transitions) -> float:
    """KL divergence between the expert and the current policy.
    A Stablebaseline3-format expert policy is required.

    :return: The average KL divergence between the the expert policy and the current policy
    """
    obs = copy.deepcopy(transitions.obs)
    acts = copy.deepcopy(transitions.acts)

    obs_th = torch.as_tensor(obs, device=device)
    acts_th = torch.as_tensor(acts, device=device)

    # 确保模型的权重在同一设备上
    sqil_trainer.policy.to(device)
    expert.policy.to(device)

    target_values, target_log_prob, target_entropy = expert.policy.evaluate_actions(obs_th, acts_th)

    with torch.no_grad():
        q_values_sqil = sqil_trainer.policy.q_net(obs_th)
        probs_sqil = F.softmax(q_values_sqil, dim=1)

        # 选择与动作 acts_th 对应的概率
    probs_sqil_selected = probs_sqil.gather(1, acts_th.long().unsqueeze(-1)).squeeze(-1)

    kl_div = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - probs_sqil_selected.log()))
    return (float(kl_div))

sqil_trainer = SQIL(
    venv=env,
    demonstrations=transitions,
    policy="CnnPolicy",
    rl_kwargs={"buffer_size": 10000}
)
total_timesteps = 128000  # 总训练步数
eval_interval =128  # 每隔多少步测试一次

log = 0
# 训练和测试循环
for timestep in tqdm.tqdm(range(0, total_timesteps, eval_interval)):
    # 训练模型

    sqil_trainer.train(total_timesteps=eval_interval)
    log += 1

    kl = expert_kl(sqil_trainer, expert, transitions)

    evaluate = _evaluate_policy(sqil_trainer, env)
    writer.add_scalar("Result/distance", kl, log)
    writer.add_scalar("Result/reward", evaluate, log)
    print(f"Round {timestep}: KL Divergence = {kl}, Mean Reward = {evaluate}")