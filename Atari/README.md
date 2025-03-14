# Atari Environment - TRRL

This project provides training and evaluation code for Atari environments using TRRL (Trust Region Reinforcement Learning) algorithms.

---

## 📦 Project Structure

```
Atari/
├── expert_data/           # Pre-collected expert trajectories and datasets
├── AIRL.py                # Adversarial Inverse Reinforcement Learning (AIRL) implementation
├── BC.py                  # Behavioral Cloning (BC) implementation
├── GAIL.py                # Generative Adversarial Imitation Learning (GAIL)
├── SQIL.py                # Soft Q Imitation Learning (SQIL)
├── firl.py                # Feature-based Inverse Reinforcement Learning (FIRL)
├── reward_function.py     # Reward function definitions
├── rollouts.py            # Environment rollouts
├── trrl.py                # TRRL algorithm core
├── arguments.py           # Argument parser configuration
└── main.py                # Main training entry point
```

---

## 🚀 How to Run Atari Training

You can run the training script by specifying the task (Atari environment) name using `--env_name` argument.

### Example Command:
```bash
python main.py --env_name=PongNoFrameskip-v4
```

This command will:
- Train the model on the **Pong** Atari environment.
- Load expert trajectories from `expert_data/transitions_PongNoFrameskip-v4.npy`.

---

## ✅ Supported Tasks

| Task Name                  | Description          |
|----------------------------|----------------------|
| PongNoFrameskip-v4         | Pong Atari game      |
| SpaceInvadersNoFrameskip-v4| Space Invaders Atari game |
| QbertNoFrameskip-v4        | Q*bert Atari game    |
| BreakoutNoFrameskip-v4     | Breakout Atari game  |
| CartPole-v1                | CartPole (classic control) |
| FrozenLake-v1              | Frozen Lake (classic control) |

---

## ⚙️ Command Line Arguments

| Argument         | Description                           | Example                           |
|------------------|---------------------------------------|-----------------------------------|
| `--env_name`     | The environment/task to run           | `--env_name=PongNoFrameskip-v4`   |
| `--algo`         | Algorithm to use (BC, SQIL, AIRL, etc.) | `--algo=SQIL`                    |
| `--num_episodes` | Number of episodes for training       | `--num_episodes=5000`            |

You can combine these flags depending on your needs.

---

## 💾 Expert Data

Expert datasets are stored in:
```
Atari/expert_data/
```
Each task has:
- `transitions_<TaskName>.npy` files (state-action trajectories)
- Pre-collected expert rollouts in `.zip` format

---

## 🔧 Dependencies

Install required Python packages:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install gym numpy torch
```

---

## ✨ Example Training Runs

### Run Behavioral Cloning (BC) on Pong:
```bash
python BC.py --env_name=PongNoFrameskip-v4
```

### Run SQIL on Space Invaders:
```bash
python SQIL.py --env_name=SpaceInvadersNoFrameskip-v4
```

### Run AIRL on Q*bert:
```bash
python AIRL.py --env_name=QbertNoFrameskip-v4
```

---

## 📂 Output and Logs
Training results and logs will be saved automatically under:
```
logs/
```

---

## 📝 Notes
- Make sure the `expert_data` folder contains the correct trajectory files for your selected task.
- For larger models or datasets, consider using **Git LFS**.

---
