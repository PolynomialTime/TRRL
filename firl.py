"""f-IRL.
"""
import torch
import gym
import numpy as np
import copy

from torch.autograd import grad

from typing import Optional

from imitation.algorithms import base
from imitation.algorithms import base
from imitation.data import types
from imitation.data import rollout
from imitation.util import logger
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util.util import make_vec_env
from imitation.data import rollout

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common import policies, vec_env, evaluation, preprocessing

from reward_function import RwdFromRwdNetFIRL

from tqdm import tqdm

import datetime
import os
import torch.utils.tensorboard as tb


import arguments

class FIRL(base.DemonstrationAlgorithm[types.Transitions]):
    def __init__(
            self,
            venv,
            expert_policy,
            demonstrations, 
            trajectory_length=64,
            batch_size=16,
            custom_logger: Optional[logger.HierarchicalLogger] = None,
            device:torch.device = torch.device("cpu"),
            allow_variable_horizon:bool = False, 
            divergence="kl",
            reward_lr=1e-3,
            discriminator_lr=1e-3,
            ent_coef = 0.01,
            discount = 0.99,
            arglist = None,
            **kwargs
    ): 
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )
        self.arglist = arglist
        self.demonstrations = demonstrations
        self.expert_policy = expert_policy
        self.env = venv
        self._policy = None
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size
        self.divergence = divergence
        self.current_iteration = 0
        self.n_policy_updates_per_round = 100_000
        self.discount = discount
        self.device = device
        self.reward_lr = reward_lr
        self.discriminator_lr = discriminator_lr
        self.ent_coef = ent_coef
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(self.demonstrations.obs.shape[1], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),
        ).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=discriminator_lr
        )
        self.reward_net = torch.nn.Sequential(
            torch.nn.Linear(self.demonstrations.obs.shape[1], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        ).to(self.device)
        self.reward_optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=1e-3)
        
    @property
    def policy(self):
        return self._policy
    
    def set_demonstrations(self, demonstrations) -> None:
        self.demonstrations = demonstrations

    def train_new_policy_for_new_reward(self):
        """Update the policy to maximise the rewards under the new reward function. The PPO algorithm will be used.

        Returns:
            A gym PPO policy optimised for the current reward network
        """

        # setup an env with the reward being the current reward network

        rwd_fn = RwdFromRwdNetFIRL(rwd_net=self.reward_net)
        venv_with_cur_rwd_net = RewardVecEnvWrapper(
            venv=self.env,
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
            ent_coef=0.01,
            device=self.device
        )
        new_policy.learn(self.n_policy_updates_per_round)
        self.current_iteration += 1
        self._policy = new_policy

    def train_discriminator_imitation(self, epochs=100):
        """
        Train a discriminator D_\omega(s) to estimate the density ratio using the imitation package.

        Args:
            epochs (int): Number of training epochs.

        Returns:
            DiscriminatorNet: The trained discriminator.
        """
        # Extract expert states from demonstrations
        expert_states = self.demonstrations.obs

        optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr)
        criterion = torch.nn.BCELoss()

        # Training loop
        for epoch in range(epochs):
            # Generate policy rollouts
            rng = np.random.default_rng(0)
            trajs = rollout.generate_trajectories(
                policy=self.policy,
                venv=self.env,
                sample_until=rollout.make_sample_until(min_timesteps=None, min_episodes=512),
                rng=rng,
            )
            trajectories = rollout.flatten_trajectories(trajs)
            policy_states = trajectories.obs[: len(expert_states)]  # Match expert states count

            # Create labels: 1 for expert states, 0 for policy states
            expert_labels = torch.ones(len(expert_states), 1, device=self.device)
            policy_labels = torch.zeros(len(policy_states), 1, device=self.device)

            # Preprocess expert states
            #expert_states_preprocessed = discriminator.preprocess(expert_states)
            #policy_states_preprocessed = discriminator.preprocess(policy_states)

            # Combine data
            combined_states = np.vstack([expert_states, policy_states])
            combined_states = torch.tensor(combined_states, dtype=torch.float32, device=self.device)
            combined_labels = torch.cat([expert_labels, policy_labels], dim=0)

            # Shuffle data
            indices = np.random.permutation(len(combined_states))
            combined_states = combined_states[indices]
            combined_labels = combined_labels[indices]

            # Training the discriminator
            for _ in range(1):
                # Forward pass through the discriminator
                predictions = self.discriminator(combined_states)
                loss = criterion(predictions, combined_labels)

                # Backward pass and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    def estimate_density_ratio(self, states):
        """
        Estimate the density ratio \(\rho_E(s) / \rho_\theta(s)\) using the trained discriminator.

        Args:
            discriminator (DiscriminatorNet): Trained discriminator.
            states (np.ndarray): States for which to compute the density ratio.
            device (str): Device for computation ('cpu' or 'cuda').

        Returns:
            torch.tensor: Density ratio values for the given states.
        """
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        d_omega = self.discriminator(states_tensor)
        density_ratios = d_omega / (1 - d_omega)
        return density_ratios
    
    def update_reward_network(self, policy_trajectories):
        """
        Perform gradient descent to update the reward network parameters θ.

        Args:
            policy_trajectories (list): List of policy trajectories as `TrajectoryWithRew`.

        Returns:
            torch.nn.Module: Updated reward network.
        """
        # Collect all states from policy trajectories
        all_states = torch.tensor(
            policy_trajectories.obs,
            dtype=torch.float32,
            device=self.device,
        )
        self.train_discriminator_imitation()
        density_ratios = self.estimate_density_ratio(all_states)
        all_density_ratios = torch.tensor(density_ratios, dtype=torch.float32, device=self.device)

        T = self.demonstrations.obs.shape[0]  # Assume all trajectories have the same length
        # Initialize accumulators for expectations
        total_params = sum(p.numel() for p in self.reward_net.parameters())
        grad_r_sum = torch.zeros(total_params, device=self.device)
        grad_r_mean = torch.zeros(total_params, device=self.device)
        density_sum = 0.0

        # Compute gradient terms for the formula
        for i, s_t in enumerate(all_states):
            # Forward pass: compute reward
            r_t = self.reward_net(s_t.unsqueeze(0))  # Add batch dimension
            
            # Compute reward gradient wrt reward network parameters
            grad_r = grad(
                r_t.sum(), self.reward_net.parameters(), retain_graph=True, create_graph=True
            )
            grad_r_combined = torch.cat([g.view(-1) for g in grad_r])  # Flatten gradients

            # Density ratio: Dω(s) / (1 - Dω(s))
            d_ratio = all_density_ratios[i]
            
            # Reshape d_ratio for broadcasting
            d_ratio_expanded = d_ratio.view(1)  # Shape [1], broadcasts correctly
            
            # Accumulate terms
            grad_r_sum += -d_ratio_expanded * grad_r_combined
            density_sum += -d_ratio
            grad_r_mean += grad_r_combined

        # Normalize terms by trajectory length T
        grad_r_sum /= T
        density_sum /= T
        grad_r_mean /= T

        # Compute the gradient using the formula
        gradient = (1 / (self.ent_coef * T)) * (grad_r_sum - density_sum * grad_r_mean)

        # Update reward network parameters using optimizer
        self.reward_optimizer.zero_grad()
        start_idx = 0
        for param in self.reward_net.parameters():
            param_grad = gradient[start_idx : start_idx + param.numel()].view(param.shape)
            param.grad = param_grad
            start_idx += param.numel()
        self.reward_optimizer.step()

    @property
    def evaluate_policy(self) -> float:
        """Evalute the true expected return of the learned policy under the original environment.

        :return: The true expected return of the learning policy.
        """
        assert self.policy is not None

        mean_reward, std_reward = evaluation.evaluate_policy(model=self.policy, env=self.env)
        return mean_reward
    
    @property
    # @timeit_decorator
    def expert_kl(self) -> float:
        """KL divergence between the expert and the current policy.
        A Stablebaseline3-format expert policy is required.

        :return: The average KL divergence between the the expert policy and the current policy
        """
        assert self.policy is not None
        obs = copy.deepcopy(self.demonstrations.obs)
        acts = copy.deepcopy(self.demonstrations.acts)

        obs_th = torch.as_tensor(obs, device=self.device)
        acts_th = torch.as_tensor(acts, device=self.device)

        self.policy.policy.to(self.device)
        self.policy.policy.to(self.device)

        input_values, input_log_prob, input_entropy = self.policy.policy.evaluate_actions(obs_th, acts_th)
        target_values, target_log_prob, target_entropy = self.expert_policy.policy.evaluate_actions(obs_th, acts_th)

        kl_div = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - input_log_prob))

        return (float(kl_div))

    def train(self, n_iterations:int):

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.arglist.env_name + "/firl/" + f"logs/{current_time}"

        global writer
        writer = tb.SummaryWriter(log_dir=log_dir, flush_secs=1)
        
        for r in tqdm(range(n_iterations), desc="Training Loop"):
            # Step 1: Generate policy trajectories
            rng = np.random.default_rng(0)
            trajs = rollout.generate_trajectories(
                                                policy=self._policy,
                                                venv=self.env,
                                                sample_until=rollout.make_sample_until(min_timesteps=None, min_episodes=512),
                                                rng=rng,
            )
            trajectories = rollout.flatten_trajectories(trajs)
            trajectories = trajectories[:self.arglist.transition_truncate_len]
            #print(trajectories)

            # Step 2: Update the reward network
            self.update_reward_network(policy_trajectories=trajectories)

            # Step 3: Update the policy using PPO
            self.train_new_policy_for_new_reward()

            # Save logs
            reward = self.evaluate_policy
            distance = self.expert_kl
            
            writer.add_scalar("Result/distance", distance, r)
            writer.add_scalar("Result/reward", reward, r)

            self.logger.record("round " + str(r), 'Distance: ' + str(distance) + '. Reward: ' + str(reward))
            self.logger.dump(step=1)

            save_interval = 1
            if r % save_interval == 0:
                save_path = os.path.join(log_dir, f"reward_net_state_dict_round_{r}.pth")
                torch.save(self.reward_net.state_dict(), save_path)
                print(f"Saved reward net state dict at {save_path}")
        
        writer.close()
        


    

# Example usage
if __name__ == "__main__":

    arglist = arguments.parse_args()

    # Create environment
    rng = np.random.default_rng(0)

    env = make_vec_env(
        arglist.env_name,
        n_envs=arglist.n_env,
        rng=rng,
        parallel=True,
        max_episode_steps=500,
    )


    # Demonstrations
    expert = PPO.load(f"./expert_data/{arglist.env_name}")
    transitions = torch.load(f"./expert_data/transitions_{arglist.env_name}.npy")
    transitions = transitions[:arglist.transition_truncate_len]


    # Train reward and policy
    firl_trainer = FIRL(venv=env,
            expert_policy = expert,
            demonstrations=transitions, 
            trajectory_length=64,
            batch_size=16,
            discount=arglist.discount,
            arglist=arglist
            )
    
    firl_trainer.train(n_iterations=arglist.n_global_rounds)