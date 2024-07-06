"""Utilities for processing reward networks."""

import numpy as np
from imitation.rewards.reward_nets import RewardNet
import torch

from imitation.rewards.reward_function import RewardFn


class RwdFromRwdNet(RewardFn):
    """Use a reward network as a reward function
    """
    def __init__(self, rwd_net: RewardNet):
        """Args:
            rwd_net: The reward network to be used as a reward function
        """
        self.rwd_net = rwd_net

    def __call__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        return self.rwd_net(torch.as_tensor(state), torch.as_tensor(action), torch.as_tensor(next_state), torch.as_tensor(done)).detatch().cpu().numpy()
        # return np.zeros(shape=state.shape[:1]) # for test only


'''
class EnvRewardNetWrapper(gym.RewardWrapper):
    """Set the reward of an enviornment as the value determined by a supplied reward network
    """

    def __init__(self, env: Union[gym.Env, VecEnv], reward_net: RewardNet, is_vec_env: True):
        """
        Args:
            env: the gym enviornment
            reward_net: the reward network to be set
            is_vec_env: the input env is vectorised or not
        Returns:
        """
        self.env = env,
        self.reward_net = reward_net,
        self.is_vec_env = is_vec_env


    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.is_vec_env:
            cur_obs = self.env.unwrapped.envs[0].unwrapped.state
            if self.env.num_envs > 1:
                for idx in range(1, self.env.num_envs):
                    cur_obs = np.append([cur_obs], [self.env.unwrapped.envs[idx].unwrapped.state], axis=0)
        else:
            cur_obs = self.env.unwrapped.state
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        return next_obs, self.reward(cur_obs, action), terminated, truncated, info
    
    def reward(self, cur_obs: np.ndarray, action: np.ndarray):
        """Transforms the reward using the supplied reward network.

        Args:
            cur_obs: Current observation
            action: the action to take

        Returns:
            The transformed reward
        """
        value = self.reward_net(torch.as_tensor(cur_obs), torch.as_tensor(action)).detatch().cpu().numpy()
        return value
'''