"""f-IRL.
"""
import torch
import gym
import numpy as np

from torch.nn import BCELoss
from torch.optim import Adam
from torch.autograd import grad

from typing import Callable, Iterator, Mapping, Optional, Type, cast

from imitation.algorithms import base
from imitation.algorithms.adversarial import DiscriminatorNet
from imitation.algorithms import base
from imitation.data import types
from imitation.data import rollout
from imitation.data.rollout import generate_trajectories, flatten_trajectories
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger, networks, util
from imitation.util.logger import configure
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common import policies

from reward_function import RwdFromRwdNet, RewardNet

from tqdm import tqdm

class FIRL(base.DemonstrationAlgorithm[types.Transitions]):
    def __init__(
            self, 
            venv, 
            demonstrations, 
            logger, 
            reward_net,
            custom_logger: Optional[logger.HierarchicalLogger] = None,
            allow_variable_horizon: bool = False, 
            divergence="kl", 
            **kwargs):
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )
        self.demonstrations=demonstrations,
        self.reward_net = reward_net
        self.divergence = divergence
        self.current_iteration = 0 

    def train_new_policy_for_new_reward(self) -> policies.BasePolicy:
        """Update the policy to maximise the rewards under the new reward function. The PPO algorithm will be used.

        Returns:
            A gym PPO policy optimised for the current reward network
        """

        # setup an env with the reward being the current reward network

        rwd_fn = RwdFromRwdNet(rwd_net=self.reward_net)
        venv_with_cur_rwd_net = RewardVecEnvWrapper(
            venv=self.venv,
            reward_fn=rwd_fn
        )

        _ = venv_with_cur_rwd_net.reset()

        new_policy = PPO(
            policy=MlpPolicy,
            env=venv_with_cur_rwd_net,
            learning_rate=0.0005,
            n_epochs=5,
            gamma=self.discount,
            verbose=0,
            device='cpu'
        )

        new_policy.learn(self.n_policy_updates_per_round)

        self.current_iteration += 1

        return new_policy

    def train_discriminator_imitation(
        self,
        env: VecEnv,
        expert_trajectories: list[base.AnyTransitions],
        policy,
        batch_size=64,
        lr=1e-3,
        epochs=10,
        device="cpu",
    ):
        """
        Train a discriminator D_\omega(s) to estimate the density ratio using the imitation package.

        Args:
            env (VecEnv): Vectorized environment.
            expert_trajectories (list): Expert trajectories as a list of `TrajectoryWithRew`.
            policy: Policy for generating rollouts.
            batch_size (int): Batch size for discriminator training.
            lr (float): Learning rate for the discriminator.
            epochs (int): Number of training epochs.
            device (str): Device for computation ('cpu' or 'cuda').

        Returns:
            DiscriminatorNet: The trained discriminator.
        """
        # Extract expert states from expert trajectories
        expert_states = np.concatenate([traj.obs for traj in expert_trajectories])

        # Create a discriminator
        state_dim = env.observation_space.shape[0]
        discriminator = DiscriminatorNet(
            observation_space=env.observation_space,
            action_space=env.action_space,
            use_action=False,  # We only need state-based density ratios
        ).to(device)

        optimizer = Adam(discriminator.parameters(), lr=lr)
        criterion = BCELoss()

        # Training loop
        for epoch in range(epochs):
            # Generate policy rollouts
            policy_trajectories = rollout.generate_trajectories(policy, env, n_episodes=10)
            policy_states = np.concatenate([traj.obs for traj in policy_trajectories])

            # Create labels: 1 for expert states, 0 for policy states
            expert_labels = torch.ones(len(expert_states), 1, device=device)
            policy_labels = torch.zeros(len(policy_states), 1, device=device)

            # Combine data
            combined_states = np.vstack([expert_states, policy_states])
            combined_labels = torch.cat([expert_labels, policy_labels], dim=0)

            # Shuffle data
            indices = np.random.permutation(len(combined_states))
            combined_states = combined_states[indices]
            combined_labels = combined_labels[indices]

            # Batch training
            num_batches = len(combined_states) // batch_size
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                batch_states = torch.tensor(
                    combined_states[start:end], dtype=torch.float32, device=device
                )
                batch_labels = combined_labels[start:end]

                # Forward pass
                predictions = discriminator(batch_states)
                loss = criterion(predictions, batch_labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        return discriminator


    def estimate_density_ratio(discriminator, states, device="cpu"):
        """
        Estimate the density ratio \(\rho_E(s) / \rho_\theta(s)\) using the trained discriminator.

        Args:
            discriminator (DiscriminatorNet): Trained discriminator.
            states (np.ndarray): States for which to compute the density ratio.
            device (str): Device for computation ('cpu' or 'cuda').

        Returns:
            torch.tensor: Density ratio values for the given states.
        """
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        d_omega = discriminator(states_tensor)
        density_ratios = d_omega / (1 - d_omega)
        return density_ratios
    
    def update_reward_network(
        self,
        policy_trajectories,
        reward_net,
        density_ratios,
        optimizer,
        alpha,
        device="cpu",
    ):
        """
        Perform gradient descent to update the reward network parameters θ.

        Args:
            policy_trajectories (list): List of policy trajectories as `TrajectoryWithRew`.
            reward_net (torch.nn.Module): Reward network \( r_\theta(s) \).
            density_ratios (np.ndarray): Precomputed density ratios \( \frac{D_\omega(s)}{1 - D_\omega(s)} \).
            optimizer (torch.optim.Optimizer): Optimizer for updating \( \theta \).
            alpha (float): Learning rate scaling factor.
            device (str): Device for computation ('cpu' or 'cuda').

        Returns:
            torch.nn.Module: Updated reward network.
        """
        # Collect all states from policy trajectories
        all_states = torch.tensor(
            np.concatenate([traj.obs for traj in policy_trajectories]),
            dtype=torch.float32,
            device=device,
        )
        all_density_ratios = torch.tensor(density_ratios, dtype=torch.float32, device=device)

        T = len(policy_trajectories[0].obs)  # Assume all trajectories have the same length

        # Initialize accumulators for expectations
        grad_r_sum = torch.zeros_like(next(reward_net.parameters()), device=device)
        grad_r_mean = torch.zeros_like(next(reward_net.parameters()), device=device)
        density_sum = 0.0

        # Compute gradient terms for the formula
        for i, s_t in enumerate(all_states):
            s_t = s_t.unsqueeze(0)  # Add batch dimension
            # Forward pass: compute reward
            r_t = reward_net(s_t)
            # Compute reward gradient wrt reward network parameters
            grad_r = grad(
                r_t.sum(), reward_net.parameters(), retain_graph=True, create_graph=True
            )
            grad_r_combined = torch.cat([g.view(-1) for g in grad_r])

            # Density ratio: Dω(s) / (1 - Dω(s))
            d_ratio = all_density_ratios[i]

            # Accumulate terms
            grad_r_sum += -d_ratio * grad_r_combined
            density_sum += -d_ratio
            grad_r_mean += grad_r_combined

        # Normalize terms by trajectory length T
        grad_r_sum /= T
        density_sum /= T
        grad_r_mean /= T

        # Compute the gradient using the formula
        gradient = (1 / (alpha * T)) * (grad_r_sum - density_sum * grad_r_mean)

        # Update reward network parameters using optimizer
        optimizer.zero_grad()
        start_idx = 0
        for param in reward_net.parameters():
            param_grad = gradient[start_idx : start_idx + param.numel()].view(param.shape)
            param.grad = param_grad
            start_idx += param.numel()
        optimizer.step()

        return reward_net    

    def train(self,
              env,
              reward_net,
              policy,
              iterations,
              reward_optimizer,
              alpha,
              batch_size=100,
              device="cpu",
              trajectory_length=10):
        
        for iteration in tqdm(range(iterations), desc="Training Loop"):
        # Step 1: Generate policy trajectories
        trajectories = generate_trajectories(policy, env, n_episodes=batch_size)
        flat_trajectories = flatten_trajectories(trajectories)