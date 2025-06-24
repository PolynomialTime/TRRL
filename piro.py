"""Proximal Inverse Reward Optimiza (PIRO).

Trains a reward function whose induced policy is monotonically improved towards the expert policy.
"""
import math
import os
import time
from typing import Callable, Iterator, Mapping, Optional, Type, cast
import copy
import tqdm
import torch
import numpy as np
import gymnasium as gym
from functools import wraps
from stable_baselines3.common.evaluation import evaluate_policy
import datetime
import copy

from stable_baselines3 import PPO, SAC
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common import policies, vec_env, evaluation, preprocessing
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies   import SACPolicy

from imitation.algorithms import base as algo_base
from imitation.algorithms import base
from imitation.data import types
from imitation.util import logger, networks, util
from imitation.util.util import make_vec_env
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper

from reward_function import RwdFromRwdNet, RewardNet
import rollouts
import random

import torch.utils.tensorboard as tb


def timeit_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        temp_str = str(func.__code__)
        return result

    return wrapper


class PIRO(algo_base.DemonstrationAlgorithm[types.Transitions]):
    """Trust Region Reward Learning (PIRO).

        Trains a reward function whose induced policy is monotonically improved towards the expert policy.
    """

    def __init__(
            self,
            *,
            venv: vec_env.VecEnv,
            expert_policy: policies = None,
            demonstrations: base.AnyTransitions,
            demo_batch_size: int,
            custom_logger: Optional[logger.HierarchicalLogger] = None,
            reward_net: RewardNet,
            discount: np.float32,
            target_reward_diff: 0.005,
            target_reward_l2_norm: 0.1,
            avg_diff_coef: np.float32,
            l2_norm_coef: np.float32,
            l2_norm_upper_bound: np.float32,
            ent_coef: np.float32 = 0.01,
            rwd_opt_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
            device: torch.device = torch.device("cpu"),
            log_dir: types.AnyPath = "output/",
            allow_variable_horizon: bool = False,
            n_policy_updates_per_round=100_000,
            n_reward_updates_per_round=10,
            n_episodes_adv_fn_est=32,
            n_timesteps_adv_fn_est=64,
            observation_space=None,
            action_space=None,
            arglist=None,
            **kwargs,
    ):
        """
        Builds PIRO.

        :param venv: The vectorized environment to train on.
        :param expert_policy: The expert polocy in the form of stablebaseline3 policies. This is used to
            calculate the difference between the expert policy and the learned policy.
        :param demonstrations: Demonstrations to use for training. The input demo should be flatened.
        :param old_policy: The policy model to use for the old policy (Stable Baseline 3).
        :param demo_batch_size: The number of samples in each batch of expert data.
        :param custom_logger: Where to log to; if None (default), creates a new logger.
        :param reward_net: reward network.
        :param discount: discount factor. A value between 0 and 1.
        :param avg_diff_coef: coefficient for soft Bellman error.
        :param l2_norm_coef: coefficient for the max difference between r_new and r_old.
            In the practical algorithm, the max difference is replaced
            by an average distance for the differentiability.
        :param l2_norm_upper_bound: Upper bound for the l2 norm of the difference between current and old reward net
        :param ent_coef: coefficient for policy entropy.
        :param rwd_opt_cls: The optimizer for reward training
        :param n_policy_updates_per_round: The number of rounds for updating the policy per global round.
        :param n_reward_updates_per_round: The number of rounds for updating the reward per global round.
        :param log_dir: Directory to store TensorBoard logs, plots, etc. in.
        :param kwargs: Keyword arguments to pass to the RL algorithm constructor.

        :raises: ValueError: if `dqn_kwargs` includes a key
                `replay_buffer_class` or `replay_buffer_kwargs`.
        """
        self.arglist = arglist
        self.action_space = action_space
        self.observation_space = observation_space
        self._rwd_opt_cls = rwd_opt_cls
        self._old_policy = None
        self._old_reward_net = None
        self.ent_coef = ent_coef
        self.avg_diff_coef = avg_diff_coef
        self.l2_norm_coef = l2_norm_coef
        self.l2_norm_upper_bound = l2_norm_upper_bound
        self.target_reward_diff = target_reward_diff
        self.target_reward_l2_norm = target_reward_l2_norm
        # self.expert_state_action_density = self.est_expert_demo_state_action_density(demonstrations)        
        self.venv = venv
        self.device = device
        self._expert_policy = expert_policy
        self.demonstrations = demonstrations
        self.demo_batch_size = demo_batch_size
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )
        self._new_policy = None  # Initialize _new_policy
        self._reward_net = reward_net.to(device)
        self._rwd_opt = self._rwd_opt_cls(self._reward_net.parameters(), lr=self.arglist.lr)
        self.discount = discount
        self.n_policy_updates_per_round = n_policy_updates_per_round
        self.n_reward_updates_per_round = n_reward_updates_per_round
        self._log_dir = util.parse_path(log_dir)
        # self.logger = logger.configure(self._log_dir)
        self._global_step = 0
        self.current_iteration = 0  # 当前策略迭代次数
        self.trajectory_buffer_v = {}
        self.trajectory_buffer_q = {}
        self.MAX_BUFFER_SIZE_PER_KEY = self.arglist.buffer_size
        self.behavior_policy = None
          
    @property
    def expert_kl(self) -> float:
        """KL divergence between the expert and the current policy.
        A Stablebaseline3-format expert policy is required.

        :return: The average KL divergence between the the expert policy and the current policy
        """
        assert self._old_policy is not None
        # assert isinstance(self._old_policy.policy, policies.ActorCriticPolicy)
        # assert isinstance(self._expert_policy.policy, policies.ActorCriticPolicy)

        # copy demo data
        obs = copy.deepcopy(self.demonstrations.obs)
        acts = copy.deepcopy(self.demonstrations.acts)

        obs_th = torch.as_tensor(obs, device=self.device)
        acts_th = torch.as_tensor(acts, device=self.device)

        # move both policies to device
        self._old_policy.policy.to(self.device)
        self._expert_policy.policy.to(self.device)

        old_pol    = self._old_policy.policy
        expert_pol = self._expert_policy.policy

        if isinstance(old_pol, ActorCriticPolicy) and isinstance(expert_pol, ActorCriticPolicy):
            # for PPO
            # _, old_log_prob, _ = old_pol.evaluate_actions(obs_th, acts_th)
            # _, new_log_prob, _ = expert_pol.evaluate_actions(obs_th, acts_th)
            # kl_div = torch.mean(torch.dot(torch.exp(new_log_prob) , (new_log_prob - old_log_prob)))

            input_values, input_log_prob, input_entropy = self._old_policy.policy.evaluate_actions(obs_th, acts_th)
            target_values, target_log_prob, target_entropy = self._expert_policy.policy.evaluate_actions(obs_th, acts_th)
            #kl_div = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - input_log_prob))
            #kl_div = torch.mean(target_log_prob - input_log_prob)
            kl_div = torch.sqrt(torch.mean((target_log_prob - input_log_prob) ** 2))

            
        elif isinstance(old_pol, SACPolicy) and isinstance(expert_pol, SACPolicy):
            with torch.no_grad():
                # old policy: get distribution params ⇒ log_prob
                mean_old, log_std_old, extras_old = old_pol.actor.get_action_dist_params(obs_th)
                _, old_log_prob = old_pol.actor.action_dist.log_prob_from_params(mean_old, log_std_old, **extras_old)

                mean_new, log_std_new, extras_new = expert_pol.actor.get_action_dist_params(obs_th)
                _, new_log_prob = expert_pol.actor.action_dist.log_prob_from_params(mean_new, log_std_new, **extras_new)

            #kl_div = torch.mean(torch.dot(torch.exp(new_log_prob), (new_log_prob - old_log_prob)))
            kl_div = torch.sqrt(torch.mean(new_log_prob - old_log_prob)** 2)

        else:
            raise ValueError(
                f"Unsupported policy type: {old_pol.__class__.__name__} / {expert_pol.__class__.__name__}."
            )

        return float(kl_div)
    
    # def evaluate_real_return(self, n_episodes=10, horizon=1000, deterministic=False) -> float:
    #     """Evaluate the true expected return of the learned policy under the original environment.
    #     This version provides more control over the evaluation process.

    #     Args:
    #         n_episodes: Number of episodes to evaluate
    #         horizon: Maximum steps per episode
    #         deterministic: Whether to use deterministic action selection

    #     Returns:
    #         The average return across episodes
    #     """
    #     assert self._old_policy is not None
        
    #     returns = []
    #     env = self.venv
        
    #     for _ in range(n_episodes):
    #         obs = env.reset()
    #         episode_reward = 0
    #         for t in range(horizon):
    #             action, _ = self._old_policy.predict(obs, deterministic=deterministic)
    #             obs, rewards, dones, infos = env.step(action)
    #             episode_reward += rewards[0]  # Get the first reward since venv returns arrays
    #             if dones[0]:  # Check if the first environment is done
    #                 break
    #         returns.append(episode_reward)
        
    #     mean_return = np.mean(returns)
    #     return mean_return

    # @property
    # def evaluate_policy(self) -> float:
    #     """Original evaluate_policy now calls evaluate_real_return"""
    #     return self.evaluate_real_return()

    @property
    def evaluate_policy(self) -> float:
        """Evalute the true expected return of the learned policy under the original environment.

        :return: The true expected return of the learning policy.
        """
        assert self._old_policy is not None
        #assert isinstance(self._old_policy.policy, policies.ActorCriticPolicy)

        mean_reward, std_reward = evaluation.evaluate_policy(model=self._old_policy, env=self.venv, n_eval_episodes=20)
        return mean_reward

    def log_saving(self) -> None:
        """Save logs containing the following info:
                1. KL divergence between the expert and the current policy;
                2. Evaluations of the current policy.
        """
        # TODO
        pass

    def set_demonstrations(self, demonstrations) -> None:
        self.demonstrations = demonstrations

    def reset(self, reward_net: RewardNet = None):
        """Reset the reward network and the iteration counter.

        Args:
            reward_net: The reward network to set as.
        """
        self._reward_net = reward_net
        self._old_reward_net = None

    def est_expert_demo_state_action_density(self, demonstration: base.AnyTransitions) -> np.ndarray:
        pass

    def compute_is_weights(self, behavior_policy, new_policy, observations, actions):
        """
        Compute the importance sampling (IS) weights.

        Args:
            behavior_policy: The old policy used to generate the original trajectories.
            new_policy: The new policy that we are trying to evaluate.
            observations: The observations from the trajectory.
            actions: The actions taken in the trajectory.

        Returns:
            weights: The computed IS weights.
        """
        observations = torch.as_tensor(observations, device=self.device)
        actions      = torch.as_tensor(actions, device=self.device)

        old_pol    = behavior_policy.policy
        new_pol    = new_policy.policy

        if isinstance(old_pol, ActorCriticPolicy) and isinstance(new_pol, ActorCriticPolicy):
            # for PPO
            old_log_prob = old_pol.evaluate_actions(observations, actions)[1]
            new_log_prob = new_pol.evaluate_actions(observations, actions)[1]
        elif isinstance(old_pol, SACPolicy) and isinstance(new_pol, SACPolicy):
            # for SAC
            with torch.no_grad():
                latent_old, _ = old_pol._get_latent(observations)
                dist_old      = old_pol._get_action_dist_from_latent(latent_old)
                old_log_prob  = dist_old.log_prob(actions).sum(dim=-1)

                latent_new, _ = new_pol._get_latent(observations)
                dist_new      = new_pol._get_action_dist_from_latent(latent_new)
                new_log_prob  = dist_new.log_prob(actions).sum(dim=-1)
        else:
            raise ValueError("Unsupported policy type for IS weights")

        weights = torch.exp(new_log_prob - old_log_prob)
        return weights

    def reward_of_sample_traj_old_policy_cur_reward(self, starting_state: np.ndarray, n_timesteps: int, n_episodes: int) -> torch.Tensor:
        """Sample trajectories under the old policy and the current reward network

        Args:
            starting_state: The state that a rollout starts at.
            n_timesteps: The length of a rollout.
            n_episodes: The number of simulated rollouts.

        Returns:
            the averaged total return of sampled trajectories under that current reward parameter
        """
        self._old_policy.policy.to(self.device)
        self._expert_policy.policy.to(self.device)
        if isinstance(self.observation_space, gym.spaces.Discrete):
            starting_s = starting_state.astype(int)
        else:
            starting_s = starting_state

        discounts = torch.pow(torch.ones(n_timesteps, device=self.device) * self.discount,
                              torch.arange(0, n_timesteps, device=self.device))

        sample_num = math.ceil(n_episodes / self.arglist.n_env)

        cached_rewards = []

        for ep_idx in range(sample_num):
                    # Monte Carlo: Sample a new trajectory
                    trans = rollouts.generate_transitions_new(
                        self.behavior_policy,
                        self.venv,
                        rng=np.random.default_rng(),
                        n_timesteps=n_timesteps,
                        starting_state=starting_s,
                        truncate=True,
                    )
                    for traj in trans:
                        state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(
                                                        traj.obs, traj.acts, traj.next_obs, traj.dones)
                        rwds = self._reward_net(state_th, action_th, next_state_th, done_th)
                        cached_rewards.append(torch.dot(rwds, discounts[:len(rwds)]))
        
        return torch.mean(torch.stack(cached_rewards), dim=0)


    # @timeit_decorator
    def train_new_policy_for_new_reward(self) -> policies.BasePolicy:
        """Update the policy to maximise the rewards under the new reward function. The PPO algorithm will be used.

        Returns:
            A gym PPO policy optimised for the current reward network
        """

        # setup an env with the reward being the current reward network

        rwd_fn = RwdFromRwdNet(rwd_net=self._reward_net)
        venv_with_cur_rwd_net = RewardVecEnvWrapper(
            venv=self.venv,
            reward_fn=rwd_fn
        )

        _ = venv_with_cur_rwd_net.reset()
        if self.arglist.policy_model == 'PPO':
            new_policy = PPO(
                policy=MlpPolicy,
                env=venv_with_cur_rwd_net,
                learning_rate=self.arglist.lr,
                n_epochs=self.arglist.ppo_n_epochs,
                gamma=self.discount,
                ent_coef=self.ent_coef,
                verbose=0,
                device='cpu'
            )
        elif self.arglist.policy_model == 'SAC':
            new_policy = SAC(
                policy="MlpPolicy",
                batch_size=800,
                env=venv_with_cur_rwd_net,
                learning_rate=3e-4,
                gamma=self.discount,
                ent_coef="auto",
                verbose=0,
                device='cpu',
                #learning_starts=0
            )

        new_policy.learn(self.n_policy_updates_per_round)
        self._new_policy = new_policy
        self.current_iteration += 1

        return new_policy

    # @timeit_decorator
    def update_reward(self, use_mc=False):
        """Perform a single reward update by conducting a complete pass over the demonstrations,
        optionally using provided samples. The loss is adapted from the constrained optimisation
        problem of the trust region reward learning by Lagrangian multipliers (moving the constraints
        into the objective function).

        Args:
            use_mc: Boolean flag to determine whether to use Monte Carlo for advantage function estimation.
            cur_round: The number of current round of reward-policy iteration
        Returns:
            The updated reward network
        """
        # TODO (optional): consider optimise a reward network from scratch
        # initialise the optimiser for the reward net
        # Do a complete pass on the demonstrations, i.e., sampling sufficient batches for performing GD.
        loss = torch.zeros(1).to(self.device)
        likelihood = torch.zeros(1).to(self.device)
        avg_reward_diff = torch.zeros(1).to(self.device)
        l2_norm_reward_diff = torch.zeros(1).to(self.device)

        batch_iter = self._make_reward_train_batches()

        if use_mc:
            self.trajectory_buffer_q.clear()
            self.trajectory_buffer_v.clear()

        for batch in batch_iter:
            # estimate the advantage function
            obs = batch["state"]
            acts = batch["action"]
            next_obs = batch["next_state"]
            dones = batch["done"]
            # 打印当前批次的时间步数（n_timesteps）
            #print(f"[RewardUpdate] batch n_timesteps={obs.shape[0]}")
            
            # estimated average estimated advantage function values
            discounted_agent_return = self.reward_of_sample_traj_old_policy_cur_reward(
                starting_state=obs[0],
                n_timesteps=obs.shape[0],
                n_episodes=self.arglist.n_episodes
            )
  
            state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(obs, acts, next_obs, dones)

            discounts = torch.pow(torch.ones(obs.shape[0], device=self.device) * self.discount,
                              torch.arange(0, obs.shape[0], device=self.device))
            
            expert_rewards = self._reward_net(state_th, action_th, next_state_th, done_th)

            if self._old_reward_net is None:
                reward_diff = expert_rewards - torch.ones(1).to(self.device)
            else:
                # use `predict_th` for `self._old_reward_net` as its gradient is not needed
                # in the first episode, diff=0 because the old reward equals the new one
                reward_diff = (expert_rewards - self._old_reward_net.predict_th(obs, acts, next_obs, dones).to(self.device))

            discounted_expert_return = torch.dot(expert_rewards, discounts[:len(expert_rewards)])

            # TODO (optional): calculate over all state-action pairs
            avg_reward_diff = torch.mean(reward_diff)
            l2_norm_reward_diff = torch.norm(reward_diff, p=2)            # adaptive coef adjustment paremeters
            # if avg_diff_coef (+) too high, reduce its coef
            if avg_reward_diff > self.target_reward_diff * self.arglist.target_ratio_upper:
                self.avg_diff_coef = self.avg_diff_coef * self.arglist.coef_scale_down
            elif avg_reward_diff < self.target_reward_diff * self.arglist.target_ratio_lower:
                self.avg_diff_coef = self.avg_diff_coef * self.arglist.coef_scale_up

            self.avg_diff_coef = torch.tensor(self.avg_diff_coef)
            self.avg_diff_coef = torch.clamp(self.avg_diff_coef, min=self.arglist.coef_min, max=self.arglist.coef_max)

            # if l2_norm_reward_diff (-) too high, increase its coef
            if l2_norm_reward_diff > self.target_reward_l2_norm:
                self.l2_norm_coef = self.l2_norm_coef * self.arglist.l2_coef_scale_up
            elif l2_norm_reward_diff < self.target_reward_l2_norm:
                self.l2_norm_coef = self.l2_norm_coef * self.arglist.l2_coef_scale_down

            self.l2_norm_coef = torch.tensor(self.l2_norm_coef)
            self.l2_norm_coef = torch.clamp(self.l2_norm_coef, min=self.arglist.coef_min, max=self.arglist.coef_max)

            # loss backward
            likelihood = discounted_agent_return - discounted_expert_return
            loss = torch.abs(likelihood) + self.l2_norm_coef * l2_norm_reward_diff #+ self.avg_diff_coef * avg_reward_diff 

            self._rwd_opt.zero_grad()
            loss.backward()
            self._rwd_opt.step()

            writer.add_scalar("Batch/loss", loss.item(), self._global_step)
            writer.add_scalar("Batch/likelihood", likelihood.item(), self._global_step)
            writer.add_scalar("Batch/avg_reward_diff", avg_reward_diff.item(), self._global_step)
            writer.add_scalar("Batch/l2_norm_reward_diff", l2_norm_reward_diff.item(), self._global_step)

            self._global_step += 1

        writer.add_scalar("Update_Reward/loss", loss.item(), self._global_step)
        writer.add_scalar("Update_Reward/likelihood", likelihood.item(), self._global_step)
        writer.add_scalar("Update_Reward/avg_reward_diff", avg_reward_diff.item(), self._global_step)
        writer.add_scalar("Update_Reward/l2_norm_reward_diff", l2_norm_reward_diff.item(), self._global_step)

    # @timeit_decorator
    def train(self, n_rounds: int, callback: Optional[Callable[[int], None]] = None):
        """
        Args:
            n_rounds: An upper bound on the iterations of training.
            callback: A function called at the end of every round which takes in a
                single argument, the round number.
        """        # TODO (optional): Make the initial reward net oupput <= 1 
        # Iteratively train a reward function and the induced policy.
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.arglist.save_results_dir + f"logs/{current_time}"

        global writer
        writer = tb.SummaryWriter(log_dir=log_dir, flush_secs=1)

        #print("n_policy_updates_per_round:", self.n_policy_updates_per_round)
        #print("n_reward_updates_per_round:", self.n_reward_updates_per_round)

        save_interval = 1
        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            # Update the policy as the one optimal for the updated reward.
            start_time = time.time()

            self._old_policy = self.train_new_policy_for_new_reward()

            end_time = time.time()
            #print("train_ppo_time=", end_time - start_time)

            # Determine whether to use Monte Carlo or Importance Sampling and 
            # update the bahevior policy (the policy used to generate trajectories) as the current policy
            use_mc = (r % self.arglist.mc_interval == 0)
            if use_mc is True:
                self.behavior_policy = copy.copy(self._old_policy)

            # Update the reward network.
            for _ in range(self.n_reward_updates_per_round):
                reward_time_start = time.time()
                self.update_reward(use_mc=use_mc)
                reward_time_end = time.time()
                #print("update_reward_time=", reward_time_end - reward_time_start)

            self._old_reward_net = copy.deepcopy(self._reward_net)
            distance = self.expert_kl
            reward = self.evaluate_policy

            writer.add_scalar("Result/distance", distance, r)
            writer.add_scalar("Result/reward", reward, r)
            # 验证在 reward_net 环境下的策略回报
            rew_wrap = RewardVecEnvWrapper(self.venv, RwdFromRwdNet(self._reward_net))
            rew_policy_mean, _ = evaluation.evaluate_policy(self._old_policy, rew_wrap, n_eval_episodes=10)
            #writer.add_scalar("Result/reward_on_rwdnet", rew_policy_mean, r)
            # 在终端打印本轮指标
            #print(f"[Round {r}] distance: {distance:.4f}, reward (native): {reward:.2f}, reward_on_rwdnet: {rew_policy_mean:.2f}")
            
            self.logger.record("round " + str(r), 'Distance: ' + str(distance) + '. Reward: ' + str(reward))
            self.logger.dump(step=10)

            if r % save_interval == 0:
                save_path = os.path.join(log_dir, f"reward_net_state_dict_round_{r}.pth")
                torch.save(self._reward_net.state_dict(), save_path)
                #print(f"Saved reward net state dict at {save_path}")

            if callback:
                callback(r)

        writer.close()

    @property
    def policy(self) -> policies.BasePolicy:
        return self._old_policy

    @property
    def reward_net(self) -> RewardNet:
        return self._reward_net

    def _make_reward_train_batches(
            self,
    ) -> Iterator[Mapping[str, torch.Tensor]]:
        """Build and return training batches for the reward update.

        Args:
            expert_samples: Same as expert demonstrations.

        Returns:
            The training batch: state, action, next state, dones.
        """

        for start in range(0, self.demonstrations.obs.shape[0], self.demo_batch_size):
            end = start + self.demo_batch_size
            obs = self.demonstrations.obs[start:end]
            acts = self.demonstrations.acts[start:end]
            next_obs = self.demonstrations.next_obs[start:end]
            dones = self.demonstrations.dones[start:end]

            batch_dict = {
                "state": obs,
                "action": acts,
                "next_state": next_obs,
                "done": dones,
            }

            yield batch_dict
