"""
This is the runner of using BC (Behavioral Cloning) to infer the optimal policy.
"""
import arguments
import torch
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.bc import BC
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
import torch.utils.tensorboard as tb
import rollouts
import logging
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack
from imitation.rewards.reward_nets import BasicShapedRewardNet
from stable_baselines3.common.env_util import make_atari_env

# Remove default terminal logging
logger = logging.getLogger()
logger.handlers = []

# Parse arguments
arglist = arguments.parse_args()
rng = np.random.default_rng(0)
device = torch.device("cpu")

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
writer = tb.SummaryWriter(f"logs/{arglist.env_name}/BC", flush_secs=1)

def sample_expert_transitions(expert):
    """Sample transitions from the trained expert."""
    print("Sampling expert transitions.")
    trajs = rollouts.generate_trajectories(
        expert,
        env,
        rollouts.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=rng,
        starting_state=None,
        starting_action=None,
    )
    return rollouts.flatten_trajectories(trajs)

# Sample transitions from the expert policy
expert = PPO.load(f"./expert_data/{arglist.env_name}")
# Evaluate expert policy
mean_reward, std_reward = evaluate_policy(model=expert, env=env)
print(f"Average reward of the expert: {mean_reward}, {std_reward}.")

transitions = torch.load(f"./expert_data/transitions_{arglist.env_name}.npy")
print(f"Number of transitions in demonstrations: {transitions.obs.shape[0]}.")

transitions = transitions[:arglist.transition_truncate_len]
print(f"Number of transitions after: {transitions.obs.shape[0]}.")

# Create BC trainer
bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    batch_size=32,  # Batch size for supervised learning
    optimizer_kwargs={"lr": arglist.lr},  # Learning rate
    device=device,
    rng=rng
)

print("Starting Behavioral Cloning training.")

# Define a class to manage logging context for BC training
class BCLogger:
    def __init__(self, writer, expert, bc_trainer, env, transitions, device):
        self.writer = writer
        self.expert = expert
        self.bc_trainer = bc_trainer
        self.env = env
        self.transitions = transitions
        self.device = device
        self.epoch_idx = 0  # Initialize epoch index

    def on_epoch_end(self):
        """Log metrics at the end of each epoch."""
        obs = torch.tensor(self.transitions.obs, device=self.device)
        acts = torch.tensor(self.transitions.acts, device=self.device)

        # Compute KL divergence and evaluate reward
        kl_div = compute_kl_divergence(self.expert, self.bc_trainer.policy, obs, acts, self.device)
        mean_reward, _ = evaluate_policy(model=self.bc_trainer.policy, env=self.env, )#n_eval_episodes=10)

        # Log to TensorBoard
        self.writer.add_scalar("Result/distance", kl_div, self.epoch_idx)
        self.writer.add_scalar("Result/reward", mean_reward, self.epoch_idx)

        # Print log
        print(f"Epoch {self.epoch_idx}: KL Divergence = {kl_div:.4f}, Reward = {mean_reward:.4f}")
        self.epoch_idx += 1  # Increment epoch index

# Instantiate the logger
bc_logger = BCLogger(writer, expert, bc_trainer, env, transitions, device)

# Start training
bc_trainer.train(
    n_epochs=1000,  # Number of epochs to train
    on_epoch_end=bc_logger.on_epoch_end,  # Pass the logging method
)

# Close TensorBoard writer
writer.close()


