"""
This is the runner of using AIRL as the baseline to infer the reward functions and the optimal policy
"""
import arguments
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.adversarial import airl
from imitation.util.util import make_vec_env
import torch.utils.tensorboard as tb
#from trrl import compute_kl_divergence
import rollouts
from reward_function import BasicRewardNet
from torch.distributions import Categorical
import logging
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack
from imitation.rewards.reward_nets import BasicShapedRewardNet

class NullLogger(logging.Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle(self, record):
        # 忽略所有日志记录
        pass

def compute_kl_divergence(expert_policy, current_policy, observations, actions, device):
    """
    Computes the KL divergence between the expert policy and the current policy.

    Args:
        expert_policy: The expert policy (Stable-Baselines3 model object).
        current_policy: The current learned policy (Stable-Baselines3 model object).
        observations: Observations (numpy array or tensor).
        actions: Actions taken (numpy array or tensor).
        device: PyTorch device (e.g., 'cpu' or 'cuda').

    Returns:
        kl_divergence: The mean KL divergence.
    """
    # Convert observations and actions to tensors
    obs_th = torch.as_tensor(observations, device=device)
    acts_th = torch.as_tensor(actions, device=device)

    # Ensure both policies are on the same device
    expert_policy.policy.to(device)
    current_policy.to(device)

    # Get log probabilities from both policies
    input_values, input_log_prob, input_entropy = current_policy.evaluate_actions(obs_th, acts_th)
    target_values, target_log_prob, target_entropy = expert_policy.policy.evaluate_actions(obs_th, acts_th)

    # Compute KL divergence using TRRO's logic
    kl_divergence = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - input_log_prob)).item()

    return kl_divergence


arglist = arguments.parse_args()
rng = np.random.default_rng(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment
# env = make_vec_env(
#     arglist.env_name,
#     n_envs=arglist.n_env,
#     rng=rng,
# )
env = make_atari_env(
        arglist.env_name,      # 环境名称，例如 "PongNoFrameskip-v4"
        n_envs=arglist.n_env,  # 并行环境的数量
    )

# 添加帧堆叠 (默认堆叠最近 4 帧)
env = VecFrameStack(env, n_stack=4)
print(f"Environment set to: {arglist.env_name}")
env = VecTransposeImage(env)

# Initialize TensorBoard logger
writer = tb.SummaryWriter(log_dir=f"logs/{arglist.env_name}/AIRL", flush_secs=1)

# Sample transitions from the expert policy
expert = PPO.load(f"./expert_data/{arglist.env_name}")
# Evaluate expert policy
mean_reward, std_reward = evaluate_policy(model=expert, env=env)
print(f"Average reward of the expert: {mean_reward}, {std_reward}.")

transitions = torch.load(f"./expert_data/transitions_{arglist.env_name}.npy")
print(f"Number of transitions in demonstrations: {transitions.obs.shape[0]}.")
print("Expert observation shape:", transitions.obs.shape)


transitions = transitions[:arglist.transition_truncate_len]
print(f"Number of transitions after: {transitions.obs.shape[0]}.")

# 定义生成器算法
gen_algo = PPO(
    policy="MlpPolicy",
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=arglist.ent_coef,
    learning_rate=arglist.lr,
    gamma=arglist.discount,
    n_epochs=20,
    n_steps=128,
    device=device, 
)
# 打印生成器环境的观察值形状
sample_obs = env.reset()
print(f"Generated observation shape (after reset): {sample_obs.shape}")
rwd_net = BasicShapedRewardNet(
    env.observation_space,
    env.action_space,
)
# Create AIRL trainer
airl_trainer = airl.AIRL(
    demonstrations=transitions,
    venv=env,
    gen_algo=gen_algo,
    demo_batch_size=arglist.demo_batch_size,  
    reward_net=rwd_net,
    allow_variable_horizon=True
)

print("Starting reward learning with AIRL.")

# 设置训练的总时间步数
total_timesteps = 128_000  # 根据需要调整

# 定义回调函数以记录训练过程中的指标
def log_callback(round_idx: int):
    obs = torch.tensor(transitions.obs, device=device)  # 使用显式设备
    acts = torch.tensor(transitions.acts, device=device)
    
    kl_div = compute_kl_divergence(expert, gen_algo.policy, obs, acts, device)
    mean_reward, _ = evaluate_policy(model=gen_algo.policy, env=env,)

    writer.add_scalar("Result/distance", kl_div, round_idx)
    writer.add_scalar("Result/reward", mean_reward, round_idx)

    print(f"Round {round_idx}: KL Divergence = {kl_div}, Mean Reward = {mean_reward}")


# 开始训练
airl_trainer.train(
    total_timesteps=total_timesteps,
    callback=log_callback,
)

# 关闭 TensorBoard 写入器
writer.close()

