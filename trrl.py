"""Trust Region Reward Learning (TRRL).

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

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common import policies, vec_env, evaluation, preprocessing

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
        # print(f"Function {temp_str[-30:]} {func.__name__}  executed in {end_time - start_time} seconds")
        return result

    return wrapper


class TRRL(algo_base.DemonstrationAlgorithm[types.Transitions]):
    """Trust Region Reward Learning (TRRL).

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
        Builds TRRL.

        :param venv: The vectorized environment to train on.
        :param expert_policy: The expert polocy in the form of stablebaseline3 policies. This is used to
            calculate the difference between the expert policy and the learned policy.
        :param demonstrations: Demonstrations to use for training. The input demo should be flatened.
        :param old_policy: The policy model to use for the old policy (Stable Baseline 3).
        :param demo_batch_size: The number of samples in each batch of expert data.
        :param custom_logger: Where to log to; if None (default), creates a new logger.
        :param reward_net: reward network.
        :param discount: discount factor. A value between 0 and 1.
        :param avg_diff_coef: coefficient for `r_old - r_new`.
        :param l2_norm_coef: coefficient for the max difference between r_new and r_old.
            In the practical algorithm, the max difference is replaced
            by an average distance for the differentiability.
        :param l2_norm_upper_bound: Upper bound for the l2 norm of the difference between current and old reward net
        :param ent_coef: coefficient for policy entropy.
        :param rwd_opt_cls: The optimizer for reward training
        :param n_policy_updates_per_round: The number of rounds for updating the policy per global round.
        :param n_reward_updates_per_round: The number of rounds for updating the reward per global round.
        :param n_episodes_adv_fn_est: Number of episodes for advantage function estimation.
        :param n_timesteps_adv_fn_est: number of timesteps for advantage function estimation.
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
        self._rwd_opt = self._rwd_opt_cls(self._reward_net.parameters(), lr=0.0005)
        self.discount = discount
        self.n_policy_updates_per_round = n_policy_updates_per_round
        self.n_reward_updates_per_round = n_reward_updates_per_round
        self.n_episodes_adv_fn_est = n_episodes_adv_fn_est
        self.n_timesteps_adv_fn_est = n_timesteps_adv_fn_est
        self._log_dir = util.parse_path(log_dir)
        # self.logger = logger.configure(self._log_dir)
        self._global_step = 0
        self.current_iteration = 0  # 当前策略迭代次数
        self.trajectory_buffer_v = {}
        self.trajectory_buffer_q = {}
        self.MAX_BUFFER_SIZE_PER_KEY = 400
        self.behavior_policy = None

    @property
    # @timeit_decorator
    def expert_kl(self) -> float:
        """KL divergence between the expert and the current policy.
        A Stablebaseline3-format expert policy is required.

        :return: The average KL divergence between the the expert policy and the current policy
        """
        assert self._old_policy is not None
        assert isinstance(self._old_policy.policy, policies.ActorCriticPolicy)
        assert isinstance(self._expert_policy.policy, policies.ActorCriticPolicy)
        obs = copy.deepcopy(self.demonstrations.obs)
        acts = copy.deepcopy(self.demonstrations.acts)

        obs_th = torch.as_tensor(obs, device=self.device)
        acts_th = torch.as_tensor(acts, device=self.device)

        self._old_policy.policy.to(self.device)
        self._expert_policy.policy.to(self.device)

        input_values, input_log_prob, input_entropy = self._old_policy.policy.evaluate_actions(obs_th, acts_th)
        target_values, target_log_prob, target_entropy = self._expert_policy.policy.evaluate_actions(obs_th, acts_th)

        kl_div = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - input_log_prob))

        return (float(kl_div))

    @property
    def evaluate_policy(self) -> float:
        """Evalute the true expected return of the learned policy under the original environment.

        :return: The true expected return of the learning policy.
        """
        assert self._old_policy is not None
        assert isinstance(self._old_policy.policy, policies.ActorCriticPolicy)

        mean_reward, std_reward = evaluation.evaluate_policy(model=self._old_policy, env=self.venv)
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
        actions = torch.as_tensor(actions, device=self.device)

        old_prob = behavior_policy.policy.evaluate_actions(observations, actions)[1]
        new_prob = new_policy.policy.evaluate_actions(observations, actions)[1]

        weights = torch.exp(new_prob - old_prob)
        return weights

    # @timeit_decorator
    def est_adv_fn_old_policy_cur_reward(self, starting_state: np.ndarray, starting_action: np.ndarray,
                                         n_timesteps: int, n_episodes: int, use_mc=False) -> torch.Tensor:
        """Use Monte-Carlo or Importance Sampling to estimate the advantage function of the given state and action under the
        old policy and the current reward network

        Advantage function: A^{\pi_{old}}_{r_\theta}(s,a) = Q^{\pi_{old}}_{r_\theta}(s,a) - V^{\pi_{old}}_{r_\theta}(s,a)

        Args:
            starting_state: The state to estimate the advantage function for.
            starting_action: The action to estimate the advantage function for.
            n_timesteps: The length of a rollout.
            n_episodes: The number of simulated rollouts.
            use_mc: Boolean flag to determine whether to use Monte Carlo.

        Returns:
            the estimated value of advantage function at `starting_state` and `starting_action`
        """
        self._old_policy.policy.to(self.device)
        self._expert_policy.policy.to(self.device)

        if isinstance(self.action_space, gym.spaces.Discrete):
            starting_a = starting_action.astype(int)
        else:
            starting_a = starting_action
        if isinstance(self.observation_space, gym.spaces.Discrete):
            starting_s = starting_state.astype(int)
        else:
            starting_s = starting_state

        discounts = torch.pow(torch.ones(n_timesteps, device=self.device) * self.discount,
                              torch.arange(0, n_timesteps, device=self.device))

        sample_num = math.ceil(n_episodes / self.arglist.n_env)
        #print("n_episodes:",n_episodes,"self.n_env:",self.arglist.n_env,"sample_num:",sample_num)
        # Starting estimating Q(s,a)
        # Store trajectories staring at (s,a). We skip MC sampling for previously encounered (s,a) and will use the trajectories stored in the buffer.
        key_q = ((starting_s,) if isinstance(starting_s, (int, np.integer)) else tuple(starting_s),
                 (starting_a,) if isinstance(starting_a, (int, np.integer)) else tuple(starting_a))
        # Cache rewards of trajectories staring at (s,a) for calculating Q(s,a)
        cached_rewards_q = []

        # Monte Carlo
        if use_mc:
            if key_q not in self.trajectory_buffer_q:
                self.trajectory_buffer_q[key_q] = []
                # print("starting_s=", starting_s[0], "starting_a=", starting_a[0])
                for ep_idx in range(sample_num):
                    # Monte Carlo: Sample a new trajectory
                    trans = rollouts.generate_transitions_new(
                        self.behavior_policy,
                        self.venv,
                        rng=np.random.default_rng(0),
                        n_timesteps=n_timesteps,
                        starting_state=starting_s,
                        starting_action=starting_a,
                        truncate=True,
                    )
                    for temp_tran in trans:
                        if np.array_equal(temp_tran.obs[0],starting_s) and np.array_equal(temp_tran.acts[0],starting_a):
                            self.trajectory_buffer_q[key_q].append(temp_tran)

                            if len(self.trajectory_buffer_q[key_q]) > self.MAX_BUFFER_SIZE_PER_KEY:
                                self.trajectory_buffer_q[key_q].pop(0)

            for trajectory in self.trajectory_buffer_q[key_q]:
                state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(
                    trajectory.obs, trajectory.acts, trajectory.next_obs, trajectory.dones
                )
                rwds = self._reward_net(state_th, action_th, next_state_th, done_th)
                cached_rewards_q.append(torch.dot(rwds, discounts[:len(rwds)]))

        # Improtance Sampling
        else:
            for trajectory in self.trajectory_buffer_q[key_q]:
                state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(
                    trajectory.obs, trajectory.acts, trajectory.next_obs, trajectory.dones
                )
                rwds = self._reward_net(state_th, action_th, next_state_th, done_th)
                weights = self.compute_is_weights(self.behavior_policy, self._new_policy, trajectory.obs, trajectory.acts)
                cached_rewards_q.append(torch.dot(weights * rwds, discounts[:len(rwds)]))

        q = torch.mean(torch.stack(cached_rewards_q))

        # Starting estimating V(s)
        key_v = (starting_s,) if isinstance(starting_s, (int, np.integer)) else tuple(starting_s)
        cached_rewards_v = []
        # Monte Carlo
        if use_mc:
            if key_v not in self.trajectory_buffer_v:
                self.trajectory_buffer_v[key_v] = []
                for ep_idx in range(sample_num):
                    trans = rollouts.generate_transitions_new(
                        self.behavior_policy,
                        self.venv,
                        n_timesteps=n_timesteps,
                        rng=np.random.default_rng(0),
                        starting_state=starting_s,
                        starting_action=None,
                        truncate=True,
                    )
                    for temp_tran in trans:
                        if np.array_equal(temp_tran.obs[0],starting_s) :
                            self.trajectory_buffer_v[key_v].append(temp_tran)

                            if len(self.trajectory_buffer_v[key_v]) > self.MAX_BUFFER_SIZE_PER_KEY:
                                self.trajectory_buffer_v[key_v].pop(0)

            for trajectory in self.trajectory_buffer_v[key_v]:
                state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(
                    trajectory.obs, trajectory.acts, trajectory.next_obs, trajectory.dones
                )
                rwds = self._reward_net(state_th, action_th, next_state_th, done_th)
                cached_rewards_v.append(torch.dot(rwds, discounts[:len(rwds)]))

        # Improtance Sampling
        else:
            for trajectory in self.trajectory_buffer_v[key_v]:
                state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(
                    trajectory.obs, trajectory.acts, trajectory.next_obs, trajectory.dones
                )
                rwds = self._reward_net(state_th, action_th, next_state_th, done_th)
                weights = self.compute_is_weights(self.behavior_policy, self._new_policy, trajectory.obs, trajectory.acts)
                cached_rewards_v.append(torch.dot(weights * rwds, discounts[:len(rwds)]))

        v = torch.mean(torch.stack(cached_rewards_v))

        return q - v

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
        # TODO: consider optimise a reward network from scratch
        # initialise the optimiser for the reward net
        # Do a complete pass on the demonstrations, i.e., sampling sufficient batches for performing GD.
        loss = torch.zeros(1).to(self.device)
        avg_advantages = None
        avg_reward_diff = None
        l2_norm_reward_diff = None

        batch_iter = self._make_reward_train_batches()

        if use_mc:
            self.trajectory_buffer_q.clear()
            self.trajectory_buffer_v.clear()

        for batch in batch_iter:

            # start_batch = time.time()
            # estimate the advantage function
            obs = batch["state"]
            acts = batch["action"]
            next_obs = batch["next_state"]
            dones = batch["done"]

            # estimated average estimated advantage function values
            cumul_advantages = torch.zeros(1).to(self.device)

            for idx in range(obs.shape[0]):
                cumul_advantages += self.est_adv_fn_old_policy_cur_reward(starting_state=obs[idx],
                                                                          starting_action=acts[idx],
                                                                          n_timesteps=self.n_timesteps_adv_fn_est,
                                                                          n_episodes=self.n_episodes_adv_fn_est,
                                                                          use_mc=use_mc)

            avg_advantages = cumul_advantages / obs.shape[0]
            state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(obs, acts, next_obs, dones)

            if self._old_reward_net is None:
                reward_diff = self._reward_net(state_th, action_th, next_state_th, done_th) - torch.ones(1).to(
                    self.device)
                # print("self._old_reward_net is None")
            else:
                # use `predict_th` for `self._old_reward_net` as its gradient is not needed
                # in the first episode, diff=0 because the old reward equals the new one
                reward_diff = (self._reward_net(state_th, action_th, next_state_th, done_th) -
                               self._old_reward_net.predict_th(obs, acts, next_obs, dones).to(self.device))

            # TODO: two penalties should be calculated over all state-action pairs
            avg_reward_diff = torch.mean(reward_diff)
            l2_norm_reward_diff = torch.norm(reward_diff, p=2)

            print("avg_reward_diff=", avg_reward_diff, "l2_norm_reward_diff=", l2_norm_reward_diff)

            # adaptive coef adjustment paremeters
            # if avg_diff_coef (+) too high, reduce its coef
            if avg_reward_diff > self.target_reward_diff * 1.5:
                self.avg_diff_coef = self.avg_diff_coef / 2
            elif avg_reward_diff < self.target_reward_diff / 1.5:
                self.avg_diff_coef = self.avg_diff_coef * 2

            self.avg_diff_coef = torch.tensor(self.avg_diff_coef)
            self.avg_diff_coef = torch.clamp(self.avg_diff_coef, min=1e-3, max=1e2)

            # if l2_norm_reward_diff (-) too high, increase its coef
            if l2_norm_reward_diff > self.target_reward_l2_norm:
                self.l2_norm_coef = self.l2_norm_coef * 1.2
            elif l2_norm_reward_diff < self.target_reward_l2_norm:
                self.l2_norm_coef = self.l2_norm_coef / 1.2

            self.l2_norm_coef = torch.tensor(self.l2_norm_coef)
            self.l2_norm_coef = torch.clamp(self.l2_norm_coef, min=1e-3, max=1e2)

            # loss backward
            loss = - avg_advantages + self.avg_diff_coef * avg_reward_diff + self.l2_norm_coef * l2_norm_reward_diff

            # print("Batch:", self._global_step, " loss:", round(loss.item(), 5), " avg_advantages:",
            #       round(avg_advantages.item(), 5), " self.avg_diff_coef:",
            #       round(self.avg_diff_coef.item(), 5), " avg_reward_diff:", round(avg_reward_diff.item(), 5),
            #       " self.l2_norm_coef:", round(self.l2_norm_coef.item(), 5),
            #       " l2_norm_reward_diff:", round(l2_norm_reward_diff.item(), 5))

            loss = loss * (self.demo_batch_size / self.demonstrations.obs.shape[0])

            self._rwd_opt.zero_grad()
            loss.backward()
            self._rwd_opt.step()

            writer.add_scalar("Batch/loss", loss.item(), self._global_step)
            writer.add_scalar("Batch/avg_advantages", avg_advantages.item(), self._global_step)
            writer.add_scalar("Batch/avg_reward_diff", avg_reward_diff.item(), self._global_step)
            writer.add_scalar("Batch/l2_norm_reward_diff", l2_norm_reward_diff.item(), self._global_step)

            # end_batch = time.time()
            # print("batch_time=", end_batch - start_batch)

            self._global_step += 1

        writer.add_scalar("Update_Reward/loss", loss.item(), self._global_step)
        writer.add_scalar("Update_Reward/avg_advantages", avg_advantages.item(), self._global_step)
        writer.add_scalar("Update_Reward/avg_reward_diff", avg_reward_diff.item(), self._global_step)
        writer.add_scalar("Update_Reward/l2_norm_reward_diff", l2_norm_reward_diff.item(), self._global_step)

    # @timeit_decorator
    def train(self, n_rounds: int, callback: Optional[Callable[[int], None]] = None):
        """
        Args:
            n_rounds: An upper bound on the iterations of training.
            callback: A function called at the end of every round which takes in a
                single argument, the round number.
        """
        # TODO: Make the initial reward net oupput <= 1
        # Iteratively train a reward function and the induced policy.
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.arglist.env_name + "/" + f"logs/{current_time}"

        global writer
        writer = tb.SummaryWriter(log_dir=log_dir, flush_secs=1)

        print("n_policy_updates_per_round:", self.n_policy_updates_per_round)
        print("n_reward_updates_per_round:", self.n_reward_updates_per_round)

        save_interval = 1
        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            # Update the policy as the one optimal for the updated reward.
            start_time = time.time()

            self._old_policy = self.train_new_policy_for_new_reward()

            end_time = time.time()
            print("train_ppo_time=", end_time - start_time)

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
                print("update_reward_time=", reward_time_end - reward_time_start)

            self._old_reward_net = copy.deepcopy(self._reward_net)
            distance = self.expert_kl
            reward = self.evaluate_policy

            writer.add_scalar("Result/distance", distance, r)
            writer.add_scalar("Result/reward", reward, r)

            self.logger.record("round " + str(r), 'Distance: ' + str(distance) + '. Reward: ' + str(reward))
            self.logger.dump(step=10)

            if r % save_interval == 0:
                save_path = os.path.join(log_dir, f"reward_net_state_dict_round_{r}.pth")
                torch.save(self._reward_net.state_dict(), save_path)
                print(f"Saved reward net state dict at {save_path}")

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
