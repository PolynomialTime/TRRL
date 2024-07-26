"""Trust Region Reward Learning (TRRL).

Trains a reward function whose induced policy is monotonically improved towards the expert policy.
"""
import os
from typing import Callable, Iterator, Mapping, Optional, Type
import copy
import tqdm
import torch
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common import policies, vec_env

from imitation.algorithms import base as algo_base
from imitation.algorithms import base
from imitation.data import types
from imitation.util import logger, networks, util
from imitation.util.util import make_vec_env
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper

from reward_function import RwdFromRwdNet, RewardNet
import rollouts


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
        demo_minibatch_size: int,
        custom_logger: Optional[logger.HierarchicalLogger] = None,
        reward_net: RewardNet,
        discount: np.float32,
        avg_diff_coef: np.float32,
        l2_norm_coef: np.float32,
        l2_norm_upper_bound: np.float32,
        ent_coef: np.float32 = 0.01,
        rwd_opt_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        policy_step_rounds: int,
        device: torch.device = torch.device("cpu"),
        log_dir: types.AnyPath = "output/",
        allow_variable_horizon: bool = False,
        **kwargs,
    ):
        """
        Builds TRRL.

        :param venv: The vectorized environment to train on.
        :param expert_policy: The expert polocy in the form of stablebaseline3 policies. This is used to
            calculate the difference between the expert policy and the learned policy.
        :param demonstrations: Demonstrations to use for training. The input demo should be flatened.
        :param old_policy: The policy model to use for the old policy (Stable Baseline 3).
        :param demo_batch_size: The number of samples in each batch of expert data. In principle, 
            the length of a trajectory should be a factor of mini batch size, i.e., a bacth of
            complete trajectories are used in gradient descent.
        :param demo_minibatch_size: size of minibatch to calculate gradients over.
            The gradients are accumulated until the entire batch is
            processed before making an optimization step. This is
            useful in GPU training to reduce memory usage, since
            fewer examples are loaded into memory at once,
            facilitating training with larger batch sizes, but is
            generally slower. Must be a factor of `demo_batch_size`.
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
        :param policy_step_rounds: The number of rounds for updating the policy.
        :param log_dir: Directory to store TensorBoard logs, plots, etc. in.
        :param kwargs: Keyword arguments to pass to the RL algorithm constructor.

        :raises: ValueError: if `dqn_kwargs` includes a key
                `replay_buffer_class` or `replay_buffer_kwargs`.
        """
        self._rwd_opt_cls = rwd_opt_cls
        self._old_policy = None
        self._old_reward_net = None
        self.ent_coef = ent_coef
        self.avg_diff_coef = avg_diff_coef
        self.l2_norm_coef = l2_norm_coef
        self.l2_norm_upper_bound = l2_norm_upper_bound
        #self.expert_state_action_density = self.est_expert_demo_state_action_density(demonstrations)
        self.venv = venv
        self.device = device
        self._expert_policy = expert_policy
        self.demonstrations=demonstrations
        self.demo_batch_size=demo_batch_size
        self.demo_minibatch_size = demo_minibatch_size
        if self.demo_batch_size % self.demo_minibatch_size != 0:
            raise ValueError("Batch size must be a multiple of minibatch size.")
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )
        self.venv=venv
        self._reward_net=reward_net.to(device)
        self._rwd_opt = self._rwd_opt_cls(self._reward_net.parameters(), lr=0.0005)
        self.discount=discount
        self.policy_step_rounds = policy_step_rounds
        self._log_dir = util.parse_path(log_dir)
        #self.logger = logger.configure(self._log_dir)
        self._global_step = 0

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
        
        input_values, input_log_prob, input_entropy = self._old_policy.policy.evaluate_actions(obs_th, acts_th)
        target_values, target_log_prob, target_entropy = self._expert_policy.policy.evaluate_actions(obs_th, acts_th)

        kl_div = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - input_log_prob))

        return(float(kl_div))

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

        rng = np.random.default_rng(0)

        # estimate state-action value Q^{\pi_{old}}_{r_\theta}(s,a)
        q = torch.zeros(1).to(self.device)
        for ep_idx in range(n_episodes):
            env = make_vec_env(
                env_name=self.venv.unwrapped.envs[0].unwrapped.spec.id,
                n_envs=self.venv.num_envs,
                rng=rng,
                            )

            if isinstance(self.venv.unwrapped.envs[0].unwrapped.action_space, gym.spaces.Discrete):
                starting_a = starting_action.astype(int)
            else:
                starting_a = starting_action
            if isinstance(self.venv.unwrapped.envs[0].unwrapped.observation_space, gym.spaces.Discrete):
                starting_s = starting_state.astype(int)
            else:
                starting_s = starting_state

            # Generate trajectories using the old policy, with the staring state and action being those occurring in expert demonstrations.
             
            tran = rollouts.generate_transitions(
                self._old_policy,
                env,
                rng=rng,
                n_timesteps=n_timesteps,
                starting_state=starting_s,
                starting_action=starting_a,
                truncate=True,
            )
            
            state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(tran.obs, tran.acts, tran.next_obs, tran.dones)
            rwds = self._reward_net(state_th, action_th, next_state_th, done_th)
            discounts = torch.pow(torch.ones(n_timesteps)*self.discount, torch.arange(0, n_timesteps))
            q += torch.dot(rwds, discounts)

            env.close()

        # estimate state value V^{\pi_{old}}_{r_\theta}(s,a)
        v = torch.zeros(1).to(self.device)
        for ep_idx in range(n_episodes):
            env = make_vec_env(
                env_name=self.venv.unwrapped.envs[0].unwrapped.spec.id,
                n_envs=self.venv.num_envs,
                rng=rng,
                            )
            
            if isinstance(self.venv.unwrapped.envs[0].unwrapped.observation_space, gym.spaces.Discrete):
                starting_s = starting_state.astype(int)
            else:
                starting_s = starting_state


            tran = rollouts.generate_transitions(
                self._old_policy,
                env,
                n_timesteps=n_timesteps,
                rng=rng,
                starting_state=starting_s,
                starting_action=None,
                truncate=True,
            )

            state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(tran.obs, tran.acts, tran.next_obs, tran.dones)

            rwds = self._reward_net(state_th, action_th, next_state_th, done_th)
            discounts = torch.pow(torch.ones(n_timesteps)*self.discount, torch.arange(0, n_timesteps))
            v += torch.dot(rwds, discounts)

            env.close()

        return (q-v)/n_episodes

    def train_new_policy_for_new_reward(self) -> policies.BasePolicy:
        """Update the policy to maximise the rewards under the new reward function. The PPO algorithm will be used.

        Returns:
            A gym PPO policy optimised for the current reward network
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

        new_policy = PPO(
            policy=MlpPolicy,
            env=venv_with_cur_rwd_net,
            batch_size=64,
            ent_coef=self.ent_coef,
            learning_rate=0.0005,
            n_epochs=10,
            n_steps=64,
            gamma=self.discount
        )

        new_policy.learn(self.policy_step_rounds)

        venv_with_cur_rwd_net.close()
        venv.close()
        
        return new_policy
    
    def update_reward(self) -> RewardNet:
        """Update reward network by gradient decent for `n_steps` steps. The loss is adapted from the constrained optimisation problem of the
        trust region reward learning by Lagrangian multipliers (moving the constraints into the objective function).
        
        Args:
            cur_round: The number of current round of reward-policy iteration
        Returns:
            The updated reward network
        """
        # TODO: consider optimise a reward network from scratch
        # initialise the optimiser for the reward net
        self._rwd_opt.zero_grad()
        # Do a complete pass on the demonstrations, i.e., sampling sufficient mini batches for performing GD.
        batch_iter = self._make_reward_train_batches()
        for batch in batch_iter:
            # estimate the advantage function
            obs = batch["state"]
            acts = batch["action"]
            next_obs = batch["next_state"]
            dones = batch["done"]
            
            loss = torch.zeros(1).to(self.device)

            # estimated average estimated advantage function values
            cumul_advantages = torch.zeros(1).to(self.device)

            for idx in range(self.demo_minibatch_size):
                cumul_advantages += self.est_adv_fn_old_policy_cur_reward(starting_state=obs[idx], 
                                                              starting_action=acts[idx], 
                                                              n_timesteps=64, 
                                                              n_episodes=128)
                
            avg_advantages = cumul_advantages / self.demo_minibatch_size

            state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(obs, acts, next_obs, dones)

            if self._old_reward_net is None:
                reward_diff = self._reward_net(state_th, action_th, next_state_th, done_th) - torch.ones(1)
            else:
                # use `predict_th` for `self._old_reward_net` as its gradient is not needed
                reward_diff = self._reward_net(state_th, action_th, next_state_th, done_th) - self._old_reward_net.predict_th(obs, acts, next_obs, dones)
            
            # TODO: two penalties should be calculated over all state-action pairs
            avg_reward_diff = torch.mean(reward_diff)
            
            l2_norm_reward_diff = torch.norm(reward_diff, p=2)
            
            loss = avg_advantages + self.avg_diff_coef * avg_reward_diff - self.l2_norm_coef * l2_norm_reward_diff \
                + self.l2_norm_upper_bound 
            loss = - loss * (self.demo_minibatch_size / self.demo_batch_size)
            loss.backward()

        self._global_step += 1
        return self._reward_net
    
    def train(self, n_rounds: int, callback: Optional[Callable[[int], None]] = None):
        """
        Args:
            n_rounds: An upper bound on the iterations of training.
            callback: A function called at the end of every round which takes in a
                single argument, the round number.
        """
        # TODO: Make the initial reward net oupput <= 1 
        # Iteratively train a reward function and the induced policy.
        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            # Update the policy as the one optimal for the updated reward.
            self._old_policy = self.train_new_policy_for_new_reward()
            # Update the reward network.
            self._reward_net = self.update_reward()
            self._old_reward_net = copy.deepcopy(self._reward_net)
            self.logger.record("round " + str(r), self.expert_kl())
            self.logger.dump(step=10)
            if callback:
                callback(r)

    @property
    def policy(self) -> policies.BasePolicy:
        return self._old_policy
    
    @property
    def reward_net(self) -> RewardNet:
        return self._reward_net
    
    def _make_reward_train_batches(
        self,
    ) -> Iterator[Mapping[str, torch.Tensor]]:
        """Build and return training minibatches for the reward update.

        Args:
            expert_samples: Same as expert demonstrations.

        Returns:
            The training minibatch: state, action, next state, dones.
        """

        assert isinstance(self.demonstrations.obs, np.ndarray)

        for start in range(0, self.demo_batch_size, self.demo_minibatch_size):
            end = start + self.demo_minibatch_size
            # take minibatch slice (this creates views so no memory issues)
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
