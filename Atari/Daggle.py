import os
import arguments
import torch
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
from imitation.algorithms.bc import BC
import torch.utils.tensorboard as tb
import rollouts
import logging
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import tempfile
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack
from imitation.rewards.reward_nets import BasicShapedRewardNet
from stable_baselines3.common.env_util import make_atari_env

# 创建空日志记录器实例
class NullLogger(logging.Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle(self, record):
        # 忽略所有日志记录
        pass

# Parse arguments
arglist = arguments.parse_args()
rng = np.random.default_rng(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Compute KL divergence
    kl_divergence = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - input_log_prob)).item()

    return kl_divergence

env = make_atari_env(
        arglist.env_name,      # 环境名称，例如 "PongNoFrameskip-v4"
        n_envs=arglist.n_env,  # 并行环境的数量
    )

# 添加帧堆叠 (默认堆叠最近 4 帧)
env = VecFrameStack(env, n_stack=4)
print(f"Environment set to: {arglist.env_name}")
env = VecTransposeImage(env)


# Initialize TensorBoard logger
writer = tb.SummaryWriter(f"logs/{arglist.env_name}/DAgger", flush_secs=1)

# Sample transitions from the expert policy
expert = PPO.load(f"./expert_data/{arglist.env_name}")
# Evaluate expert policy
mean_reward, std_reward = evaluate_policy(model=expert, env=env)
print(f"Average reward of the expert: {mean_reward}, {std_reward}.")

transitions = torch.load(f"./expert_data/transitions_{arglist.env_name}.npy")
print(f"Number of transitions in demonstrations: {transitions.obs.shape[0]}.")

transitions = transitions[:arglist.transition_truncate_len]
print(f"Number of transitions after: {transitions.obs.shape[0]}.")

bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    batch_size=8,  # Batch size for supervised learning
    optimizer_kwargs={"lr": arglist.lr},  # Learning rate
    device=device,
    rng=rng
)

# Define a class to manage logging context for DAgger training
class DAggerLogger:
    def __init__(self, writer, expert, dagger_trainer, env, device, transitions, bc_trainer):
        self.writer = writer
        self.expert = expert
        self.dagger_trainer = dagger_trainer
        self.env = env
        self.device = device
        self.transitions = transitions  # 添加 transitions 属性
        self.bc_trainer = bc_trainer  # 添加 bc_trainer 属性
        self.step_idx = 0  # Initialize step index

    def log_metrics(self):
        """Log metrics during training."""
        # Evaluate the policy
        obs = torch.tensor(self.transitions.obs, device=self.device)
        acts = torch.tensor(self.transitions.acts, device=self.device)

        # Compute KL divergence and evaluate reward
        kl_div = compute_kl_divergence(self.expert, self.bc_trainer.policy, obs, acts, self.device)
        mean_reward, _ = evaluate_policy(model=self.dagger_trainer.bc_trainer.policy, env=self.env)

        # Log to TensorBoard
        self.writer.add_scalar("Result/reward", mean_reward, self.step_idx)
        self.writer.add_scalar("Result/distance", kl_div, self.step_idx)

        # Print log
        print(f"Step {self.step_idx}: KL Divergence = {kl_div:.4f}, Reward = {mean_reward:.4f}")
        self.step_idx += 1  # Increment step index


# Instantiate the logger
dagger_logger = DAggerLogger(
    writer=writer,
    expert=expert,
    dagger_trainer=None,  # Will initialize inside the with block
    env=env,
    device=device,
    transitions=transitions,  # 传入 transitions
    bc_trainer=bc_trainer     # 传入 bc_trainer
)

# Using tempfile.TemporaryDirectory to handle the scratch directory
with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(f"Using temporary directory: {tmpdir}")

    # Initialize DAgger trainer
    dagger_trainer = SimpleDAggerTrainer(
        venv=env,
        scratch_dir=tmpdir,
        expert_policy=expert.policy,
        bc_trainer=bc_trainer,
        rng=rng,
    )

    print("Starting DAgger training.")  

    # Train in steps with logging
    total_steps = 10_0  # Total steps for training
    log_frequency = 1  # Log every 1000 steps

    for step in range(0, total_steps, log_frequency):
        dagger_trainer.train(total_timesteps=10, 
                             bc_train_kwargs={
            'n_batches': 1,  # 固定训练批次数量
            #'n_epochs': 1,    # 仅遍历数据集 1 次
        }  )  # Train 1000 timesteps
        dagger_logger.dagger_trainer = dagger_trainer  # Assign the trainer to the logger
        dagger_logger.log_metrics()  # Log metrics every 1000 steps

# Close TensorBoard writer
writer.close()

