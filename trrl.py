"""Trust Region Reward Learning (TRRL).

Trains a reward function whose induced policy is monotonically improved towards the expert policy.
"""
from typing import Optional

import copy
import tqdm

import torch
import numpy as np

import dataclasses

from stable_baselines3 import SAC
from stable_baselines3.common import policies, vec_env, distributions
from stable_baselines3.sac import policies as sac_policies

from imitation.algorithms import base as algo_base
from imitation.algorithms import base

from imitation.data import rollout, types

from imitation.util import logger, util
from imitation.util.util import make_vec_env

from imitation.rewards import reward_nets
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper

from reward_function import RwdFromRwdNet

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
        demo_minibatch_size: int,
        custom_logger: Optional[logger.HierarchicalLogger] = None,
        reward_net: reward_nets.RewardNet,
        discount: np.float32,
        lower_bound_coef: np.float32,
        upper_bound_coef: np.float32,
        ent_coef: np.float32 = 0.01,
        rwd_opt_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        """Builds TRRL.

        Args:
            venv: The vectorized environment to train on.
            demonstrations: Demonstrations to use for training. The input demo should be flatened.
            old_policy: The policy model to use for the old policy (Stable Baseline 3).
            demo_batch_size: The number of samples in each batch of expert data. In principle, 
                the length of a trajectory should be a factor of mini batch size, i.e., a bacth of
                complete trajectories are used in gradient descent.
            demo_minibatch_size: size of minibatch to calculate gradients over.
                The gradients are accumulated until the entire batch is
                processed before making an optimization step. This is
                useful in GPU training to reduce memory usage, since
                fewer examples are loaded into memory at once,
                facilitating training with larger batch sizes, but is
                generally slower. Must be a factor of `demo_batch_size`.
            custom_logger: Where to log to; if None (default), creates a new logger.
            reward_net: reward network.
            discount: discount factor. A value between 0 and 1.
            lower_bound_coef: coefficient for r_old - r_new.
            upper_bound_coef: coefficient for the max difference between r_new and r_old.
                                    In the practical algorithm, the max difference is replaced
                                    by an average distance for the differentiability.
            ent_coef: coefficient for policy entropy.
            rwd_opt_cls: The optimizer for reward training
            kwargs: Keyword arguments to pass to the RL algorithm constructor.

        Raises:
            ValueError: if `dqn_kwargs` includes a key
                `replay_buffer_class` or `replay_buffer_kwargs`.
        """
        self._rwd_opt_cls = rwd_opt_cls,
        self._old_policy = old_policy,
        self._old_reward_net = None,
        self.ent_coef = ent_coef,
        self.lower_bound_coef = lower_bound_coef,
        self.upper_bound_coef = upper_bound_coef,
        #self.expert_state_action_density = self.est_expert_demo_state_action_density(demonstrations)
        self.venv = venv,
        self.device = device,
        self.demonstrations=demonstrations,
        self.demo_batch_size=demo_batch_size,
        self.demo_minibatch_size = demo_minibatch_size
        if self.demo_batch_size % self.demo_minibatch_size != 0:
            raise ValueError("Batch size must be a multiple of minibatch size.")
        self.venv=venv,
        self._reward_net=reward_net.to(device),
        self._rwd_opt_cls=rwd_opt_cls,
        self._rwd_opt = self._rwd_opt_cls(self._reward_net.parameters())
        self.discount=discount,
        self.custom_logger=custom_logger


    def reset(self, reward_net: reward_nets.RewardNet = None):
        """Reset the reward network and the iteration counter.
        
        Args:
            reward_net: The reward network to set as.
        """
        self._reward_net = reward_net

    def est_expert_demo_state_action_density(self, demonstration: base.AnyTransitions) -> np.ndarray:
        pass

    def est_adv_fn_old_policy_cur_reward(self, starting_state: np.ndarray, starting_action: np.ndarray,
                                      n_timesteps: int, n_episodes: int) -> torch.Tensor:
        """Use Monte-Carlo simulation to estimation the advantage function of the given state and action under the
        old policy and the current reward network

        Advantage function: A^{\pi_{old}}_{r_\theta}(s,a) = Q^{\pi_{old}}_{r_\theta}(s,a) - V^{\pi_{old}}_{r_\theta}(s,a)

        Args:
            starting_state: The state to estimate the advantage function for.
            starting_action: The action to estimate the advantage function for.
            n_timesteps: The length of a rollout.
            n_episodes: The number of simulated rollouts.

        Returns:
            the estimated value of advantage function at `starting_state` and `starting_action`
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

            obs, acts, infos, next_obs, dones, rews = rollout.generate_trajectories(
                self._old_policy,
                env,
                n_timesteps=n_timesteps,
                rng=rng,
                starting_state=starting_state,
                starting_action=starting_action,
                #truncate=True,
            )

            for time_idx in range(n_timesteps):
                state = torch.as_tensor(obs[time_idx], self.device),
                action = torch.as_tensor(acts[time_idx], self.device),
                next_state = torch.as_tensor(next_obs[time_idx], self.device),
                done = torch.as_tensor(dones[time_idx], self.device),
                q += torch.pow(torch.as_tensor(self.discount), time_idx) * (self._reward_net(state, action, next_state, done) + self.ent_coef *
                                                                             self._get_log_policy_act_prob(obs_th=state, acts_th=action))
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

            obs, acts, infos, next_obs, dones = rollout.generate_trajectories(
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
                v += torch.pow(torch.as_tensor(self.discount), time_idx) * (self._reward_net(state, action, next_state, done) + self.ent_coef *
                                                                             self._get_log_policy_act_prob(obs_th=state, acts_th=action))
            env.close()

        return (q-v)/n_episodes

    def train_new_policy_for_new_reward(self) -> policies.BasePolicy:
        """Update the policy to maximise the rewards under the new reward function. The SAC algorithm will be used.

        Returns:
            A gym SAC policy optimised for the current reward network
        """
        rng = np.random.default_rng(0)
        venv = make_vec_env(
            env_name=self.venv.unwrapped.envs[0].unwrapped.spec.id,
            n_envs=self.venv.num_envs,
            rng=rng,
        )

        # setup an env with the reward being the current reward network
        rwd_fn = RwdFromRwdNet(rwd_net=self._reward_net)
        venv_with_cur_rwd_net = RewardVecEnvWrapper(
            venv=venv,
            reward_fn=rwd_fn
        )

        _ = venv_with_cur_rwd_net.reset()

        new_policy = SAC(
            policy=sac_policies.SACPolicy,
            env=venv_with_cur_rwd_net,
            batch_size=64,
            ent_coef=self.ent_coef,
            learning_rate=0.0003,
        )

        venv_with_cur_rwd_net.close()
        venv.close()
        
        return new_policy
    
    def update_reward(self, cur_round: int) -> reward_nets.RewardNet:
        """Update reward network by gradient decent for `n_steps` steps. The loss is adapted from the constrained optimisation problem of the
        trust region reward learning by Lagrangian multipliers (moving the constraints into the objective function).
        
        Args:
            n_steps: The number of steps for gradient decent
            cur_round: The number of current round of reward-policy iteration
        Returns:
            The updated reward network
        """
        self._rwd_opt.zero_grad()
        # Do a complete pass on the demonstrations, i.e., sampling sufficient mini batches for performing GD.
        if cur_round == 0:
            # The initial reward should be all ONE.
            # Reward tensor shape = 1 x minibatch size 
            init_rewards = torch.zeros((1, self.demo_minibatch_size))
            #TODO: estimate the advantage function
            reward_diff = self.reward_net() - init_rewards
        else:
            #TODO: add the two penality terms to the loss
            #TODO: loss backward
            pass
        


    
    def train(self, n_rounds: int, reward_training_epochs: int, callback: Optional[Callable[[int], None]] = None):
        """
        Args:
            n_rounds: An upper bound on the iterations of training.
            reward_training_steps: An upper bound on the gradient steps for training the reward network.
            callback: A function called at the end of every round which takes in a
                single argument, the round number. Round numbers are in
                `range(total_timesteps // self.gen_train_timesteps)`.
        """
        # Iteratively train a reward function and the induced policy.
        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            # Update the reward network.
            self._reward_net = self.update_reward(cur_round=r)
            self._old_reward_net = copy.deepcopy(self._reward_net)
            # Update the policy as the one optimal for the updated reward.
            self._old_policy = self.train_new_policy_for_new_reward()

    @property
    def policy(self) -> policies.BasePolicy:
        return self._old_policy
    
    @property
    def reward_net(self) -> reward_nets.RewardNet:
        return self._reward_net
    
    def _get_log_policy_act_prob(
        self,
        obs_th: torch.Tensor,
        acts_th: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Evaluates the given actions on the given observations.

        Args:
            obs_th: A batch of observations.
            acts_th: A batch of actions.

        Returns:
            A batch of log policy action probabilities.
        """
        if isinstance(self.policy, policies.ActorCriticPolicy):
            # policies.ActorCriticPolicy has a concrete implementation of
            # evaluate_actions to generate log_policy_act_prob given obs and actions.
            _, log_policy_act_prob_th, _ = self.policy.evaluate_actions(
                obs_th,
                acts_th,
            )
        elif isinstance(self.policy, sac_policies.SACPolicy):
            gen_algo_actor = self.policy.actor
            assert gen_algo_actor is not None
            # generate log_policy_act_prob from SAC actor.
            mean_actions, log_std, _ = gen_algo_actor.get_action_dist_params(obs_th)
            assert isinstance(
                gen_algo_actor.action_dist,
                distributions.SquashedDiagGaussianDistribution,
            )  # Note: this is just a hint to mypy
            distribution = gen_algo_actor.action_dist.proba_distribution(
                mean_actions,
                log_std,
            )
            # SAC applies a squashing function to bound the actions to a finite range
            # `acts_th` need to be scaled accordingly before computing log prob.
            # Scale actions only if the policy squashes outputs.
            assert self.policy.squash_output
            scaled_acts = self.policy.scale_action(acts_th.numpy(force=True))
            scaled_acts_th = torch.as_tensor(scaled_acts, device=mean_actions.device)
            log_policy_act_prob_th = distribution.log_prob(scaled_acts_th)
        else:
            return None
        return log_policy_act_prob_th
    
    def _make_reward_train_batches(
        self,
    ) -> Iterator[Mapping[str, torch.Tensor]]:
        """Build and return training minibatches for the reward update.

        Args:
            expert_samples: Same as expert demonstrations.

        Returns:
            The training minibatch: state, action, next state, dones.
        """
        batch_size = self.demo_batch_size

        expert_samples = self.demonstrations

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            if isinstance(expert_samples[k], torch.Tensor):
                expert_samples[k] = expert_samples[k].detach().numpy()

        assert isinstance(expert_samples["obs"], np.ndarray)

        # Check dimensions.
        assert batch_size == len(expert_samples["acts"])
        assert batch_size == len(expert_samples["next_obs"])

        for start in range(0, batch_size, self.demo_minibatch_size):
            end = start + self.demo_minibatch_size
            # take minibatch slice (this creates views so no memory issues)
            expert_batch = {k: v[start:end] for k, v in expert_samples.items()}

            obs = expert_batch["obs"]
            acts = expert_batch["acts"]
            next_obs = expert_batch["next_obs"]
            dones = expert_batch["dones"]

            obs_th, acts_th, next_obs_th, dones_th = self.reward_net.preprocess(
                obs,
                acts,
                next_obs,
                dones,
            )
            batch_dict = {
                "state": obs_th,
                "action": acts_th,
                "next_state": next_obs_th,
                "done": dones_th,
            }

            return batch_dict
