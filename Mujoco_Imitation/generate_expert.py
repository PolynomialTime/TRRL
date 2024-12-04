import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO

# 设置环境名称和参数
#env_name = "Ant-v4"
#env_name = "HalfCheetah-v4"
#env_name = "Hopper-v3" #1000步不结束
env_name = "Walker2d-v3"

num_trajectories = 20
max_steps = 1000

# 训练专家模型
def train_expert(env_name, total_timesteps=200000):
    print(f"{env_name}:Training expert...")
    env = gym.make(env_name)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    print("Expert training completed!")
    return model

# 收集专家轨迹
def collect_expert_trajectories(model, env_name, num_trajectories, max_steps):
    print("Collecting expert trajectories...")
    env = gym.make(env_name)
    trajectories = []

    for i in range(num_trajectories):
        obs,info = env.reset()
        trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": []
        }
        for _ in range(max_steps):
            # 使用专家模型预测动作
            action, _ = model.predict(obs)
            next_obs, reward, done, truncated, info= env.step(action)

            # 记录轨迹
            trajectory["states"].append(obs)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["next_states"].append(next_obs)
            trajectory["dones"].append(done)

            obs = next_obs
            if done:
                break

        trajectories.append(trajectory)

    print(f"Collected {len(trajectories)} trajectories!")
    return trajectories

# 保存轨迹
def save_trajectories(trajectories):
    torch.save(trajectories, f"./Expert/{env_name}.pt")
    print("Trajectories saved successfully!")

# 主流程
if __name__ == "__main__":
    # 训练专家模型
    expert_model = train_expert(env_name, total_timesteps=200000)

    # 收集专家轨迹
    expert_trajectories = collect_expert_trajectories(
        expert_model, env_name, num_trajectories, max_steps
    )
    print(expert_trajectories[:5])
    # 保存轨迹到文件
    save_trajectories(save_trajectories)
