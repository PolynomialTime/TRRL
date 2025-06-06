# test_policy_help.py

from stable_baselines3 import SAC
import gymnasium as gym

# Create a Gymnasium environment
env = gym.make("Pendulum-v1")

# Initialize a SAC model
model = SAC("MlpPolicy", env, verbose=0)

# Get the policy object
policy = model.policy

# Print the full help text for SACPolicy
help(policy)
