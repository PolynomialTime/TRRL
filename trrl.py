"""Trust Region Reward Learning (TRRL).

Trains a reward function whose induced policy is monotonically improved towards the expert policy.
"""
from typing import Optional

import torch
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common import base_class, policies, vec_env
from stable_baselines3.sac import policies as sac_policies
sac_policies.SACPolicy

from gymnasium.wrappers.transform_reward import TransformReward

from imitation.algorithms import base as algo_base
from imitation.algorithms.adversarial import common
from imitation.algorithms import base

from imitation.data import rollout, types
from imitation.data.wrappers import RolloutInfoWrapper

from imitation.util import logger, util
from imitation.util.util import make_vec_env

from imitation.rewards import reward_nets
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.rewards.reward_nets import BasicRewardNet

from typing import Callable, Iterable, Iterator, Mapping, Optional, Type, overload


class TRRL(algo_base.DemonstrationAlgorithm[types.Transitions]):
    """Trust Region Reward Learning (TRRL).

        Trains a reward function whose induced policy is monotonically improved towards the expert policy.
    """

    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        demonstrations: base.AnyTransitions,
        old_policy: policies.BasePolicy,
        demo_batch_size: int,
        custom_logger: Optional[logger.HierarchicalLogger] = None,
        reward_net: reward_nets.RewardNet,
        discount: np.float32,
        coeff_for_lower_bound: np.float32,
        coeff_for_upper_bound: np.float32,
        rwd_opt_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        """Builds TRRL.

        Args:
            venv: The vectorized environment to train on.
            demonstrations: Demonstrations to use for training.
            old_policy: The policy model to use for the old policy (Stable Baseline 3).
            custom_logger: Where to log to; if None (default), creates a new logger.
            reward_net: reward network.
            discount: discount factor. A value between 0 and 1.
            coeff_for_lower_bound: coefficient for r_old - r_new,
            coeff_for_upper_bound: coefficient for the max difference between r_new and r_old.
                                    In the practical algorithm, the max difference is replaced
                                    by an average distance for the differentiability.
            rwd_opt_cls: The optimizer for reward training
            kwargs: Keyword arguments to pass to the RL algorithm constructor.

        Raises:
            ValueError: if `dqn_kwargs` includes a key
                `replay_buffer_class` or `replay_buffer_kwargs`.
        """
        self._rwd_opt_cls = rwd_opt_cls,
        self._old_policy = old_policy,
        self.coeff_for_lower_bound = coeff_for_lower_bound,
        self.coeff_for_upper_bound = coeff_for_upper_bound,
        #self.expert_state_action_density = self.est_expert_demo_state_action_density(demonstrations)
        self.venv = venv,
        self.device = device,
        self.demonstrations=demonstrations,
        self.demo_batch_size=demo_batch_size,
        self.venv=venv,
        self._reward_net=reward_net.to(device),
        self._rwd_opt_cls=rwd_opt_cls,
        self._opt = self._rwd_opt_cls(self._reward_net.parameters())
        self.discount=discount,
        self.custom_logger=custom_logger


    def init_reward_func(self) -> np.ndarray:
        """
        The initial reward should be all ONE.
        """
        return

    def est_expert_demo_state_action_density(self, demonstration: base.AnyTransitions) -> np.ndarray:
        pass

    def est_adv_old_policy_cur_reward(self, starting_state: np.ndarray, starting_action: np.ndarray,
                                      n_timesteps: int, n_episodes: int) -> torch.Tensor:
        """
        Use Monte-Carlo simulation to estimation the advantage function of the given state and action under the
        old policy and the current reward network

        Advantage function: A^{\pi_{old}}_{r_\theta}(s,a) = Q^{\pi_{old}}_{r_\theta}(s,a) - V^{\pi_{old}}_{r_\theta}(s,a)

        Args:
            starting_state: The state to estimate the advantage function for.
            starting_action: The action to estimate the advantage function for.
            n_timesteps: The length of a rollout.
            n_episodes: The number of simulated rollouts.
        """

        # Generate trajectories using the old policy, with the staring state and action being those occurring in expert
        # demonstrations.

        #demo_obs = self.demonstrations.obs
        #demo_acts = self.demonstrations.acts

        rng = np.random.default_rng(0)

        # estimate state-action value Q^{\pi_{old}}_{r_\theta}(s,a)
        q = torch.zeros(1)
        for ep_idx in range(n_episodes):
            env = make_vec_env(
                env_name=self.venv.unwrapped.envs[0].unwrapped.spec.id,
                n_envs=self.venv.num_envs,
                rng=rng,
                # post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
                            )

            obs, acts, infos, next_obs, dones, rews = rollout.generate_transitions(
                self._old_policy,
                env,
                n_timesteps=n_timesteps,
                rng=rng,
                starting_state=starting_state,
                starting_action=starting_action,
                truncate=True,
            )

            for time_idx in range(n_timesteps):
                state = torch.as_tensor(obs[time_idx], self.device),
                action = torch.as_tensor(acts[time_idx], self.device),
                next_state = torch.as_tensor(next_obs[time_idx], self.device),
                done = torch.as_tensor(dones[time_idx], self.device),
                q += torch.pow(torch.as_tensor(self.discount), time_idx) * self._reward_net(state, action, next_state, done)
            env.close()

        # estimate state value V^{\pi_{old}}_{r_\theta}(s,a)
        v = torch.zeros(1)
        for ep_idx in range(n_episodes):
            env = make_vec_env(
                env_name=self.venv.unwrapped.envs[0].unwrapped.spec.id,
                n_envs=self.venv.num_envs,
                rng=rng,
                # post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
                            )

            obs, acts, infos, next_obs, dones, rews = rollout.generate_transitions(
                self._old_policy,
                env,
                n_timesteps=n_timesteps,
                rng=rng,
                starting_state=starting_state,
                starting_action=None,
                truncate=True,
            )

            for time_idx in range(n_timesteps):
                state = torch.as_tensor(obs[time_idx], self.device),
                action = torch.as_tensor(acts[time_idx], self.device),
                next_state = torch.as_tensor(next_obs[time_idx], self.device),
                done = torch.as_tensor(dones[time_idx], self.device),
                v += torch.pow(torch.as_tensor(self.discount), time_idx) * self._reward_net(state, action, next_state, done)
            env.close()

        return (q-v)/n_episodes

    def train_new_policy_under_new_reward(self) -> policies.BasePolicy:
        """
        Update the policy to maximise the rewards under the new reward function. The SAC algorithm will be used.
        """
        rng = np.random.default_rng(0)
        env = make_vec_env(
            env_name=self.venv.unwrapped.envs[0].unwrapped.spec.id,
            n_envs=self.venv.num_envs,
            rng=rng,
        )
        env = TransformReward(env=env, lambda r:)
        new_policy = SAC(
            policy=sac_policies.SACPolicy,
            env=self.venv,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
        )


        pass
    
    def train(self, total_timesteps, callback: Optional[Callable[[int], None]] = None):
        """
        Args:
            total_timesteps: An upper bound on the number of transitions to sample
                from the environment during training.
            callback: A function called at the end of every round which takes in a
                single argument, the round number. Round numbers are in
                `range(total_timesteps // self.gen_train_timesteps)`.
        """
        pass

    @property
    def policy(self) -> policies.BasePolicy:
        return self._old_policy