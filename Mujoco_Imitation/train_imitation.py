from utils import system, collect, logger, eval
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.bc import BC
from imitation.algorithms.sqil import SQIL
from imitation.algorithms.dagger import DAggerTrainer
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.types import Trajectory
from scipy.stats import entropy
from ruamel.yaml import YAML
import sys, os, time
from imitation.policies.serialize import load_policy
import datetime
import dateutil.tz
import json, copy
import torch.utils.tensorboard as tb
from imitation.data import rollout
from stable_baselines3.ppo import MlpPolicy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.data import huggingface_utils
import datasets
import tempfile
from imitation.algorithms.dagger import SimpleDAggerTrainer



def train_algorithm(algorithm, env_name, expert_data, device="cpu", total_timesteps=200000):
    """
    根据指定的算法训练模型。
    """
    if env_name=="Ant-v4":
        expert_policy = "ppo-seals-Ant-v1"
        expert_data ="HumanCompatibleAI/ppo-seals-Ant-v1"
    elif env_name == "HalfCheetah-v4":
        expert_policy = "ppo-seals-HalfCheetah-v1"
        expert_data = "HumanCompatibleAI/ppo-seals-HalfCheetah-v1"
    elif env_name == "Hopper-v3":
        expert_policy= "ppo-seals-Hopper-v1"
        expert_data ="HumanCompatibleAI/ppo-seals-Hopper-v1"
    elif env_name == "Walker2d-v3":
        expert_policy = "ppo-seals-Walker2d-v1"
        expert_data = "HumanCompatibleAI/ppo-seals-Walker2d-v1"
    else:
        return None

    rng = np.random.default_rng(seed)

    env = make_vec_env(
        env_name,
        rng=rng,
        n_envs=8,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
    )

    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name=expert_policy,
        venv=env,
    )

    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=np.random.default_rng(seed),
    )
    print("expert sample:", rollouts[:1])

    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0004,
        gamma=0.95,
        n_epochs=5,
        clip_range=0.1,
        vf_coef=0.1,
        seed=seed,
    )
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    agent = None
    reward = None
        # 根据算法选择对应实现
    if algorithm == "GAIL":
        gail_trainer = GAIL(
            demonstrations=rollouts,
            demo_batch_size=1024,
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=8,
            venv=env,
            gen_algo=learner,
            reward_net=reward_net,
        )

        # train the learner and evaluate again
        gail_trainer.train(20000)  # Train for 800_000 steps to match expert.
        env.seed(seed)
        reward, _ = evaluate_policy( learner, env, 100, return_episode_rewards=True)

        agent = gail_trainer.gen_algo

    elif algorithm == "AIRL":
        #
        # learner = PPO(
        #     env=env,
        #     policy=MlpPolicy,
        #     batch_size=64,
        #     ent_coef=0.0,
        #     learning_rate=0.0005,
        #     gamma=0.95,
        #     clip_range=0.1,
        #     vf_coef=0.1,
        #     n_epochs=5,
        #     seed=seed,
        # )

        airl_trainer = AIRL(
            demonstrations=rollouts,
            demo_batch_size=2048,
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=16,
            venv=env,
            gen_algo=learner,
            reward_net=reward_net,
        )

        airl_trainer.train(20000)  # Train for 2_000_000 steps to match expert.
        env.seed(seed)
        reward, _ = evaluate_policy(learner, env, 100, return_episode_rewards=True)

        agent = airl_trainer.gen_algo

    elif algorithm == "BC":
        # env = make_vec_env(
        #     env_name,
        #     rng=rng,
        #     n_envs=1,
        #     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
        # )

        transitions = rollout.flatten_trajectories(rollouts)

        bc_trainer = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            rng=rng,
        )
        bc_trainer.train(n_epochs=100)
        reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
        agent = bc_trainer.policy

    elif algorithm == "SQIL":

        sqil_trainer = SQIL(
            venv=DummyVecEnv([lambda: gym.make(env_name)]),
            demonstrations=rollouts,
            policy="MlpPolicy",
        )
        # Hint: set to 1_000_000 to match the expert performance.
        sqil_trainer.train(total_timesteps=1_000)
        reward, _ = evaluate_policy(sqil_trainer.policy, sqil_trainer.venv, 10)
        agent = sqil_trainer.policy

    elif algorithm == "Dagger":

        bc_trainer = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            rng=rng,
        )
        with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
            print(tmpdir)
            dagger_trainer = SimpleDAggerTrainer(
                venv=env,
                scratch_dir=tmpdir,
                expert_policy=expert,
                bc_trainer=bc_trainer,
                rng=rng,
            )
            dagger_trainer.train(8_000)

        reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
        agent = dagger_trainer.policy

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    obs = rollouts.obs
    acts = rollouts.acts

    input_values, input_log_prob, input_entropy = agent.evaluate_actions(obs, acts)
    target_values, target_log_prob, target_entropy = expert.policy.evaluate_actions(obs, acts)

    kl_div = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - input_log_prob))

    return kl_div,reward

if __name__ == "main":
    # 加载专家数据
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    # common parameters
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    seed = v['seed']
    num_expert_trajs = v['irl']['expert_episodes']
    n_itrs = v['irl']['n_itrs']
    algorithm = v['obj']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")

    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()

    # assumptions
    assert algorithm in ['GAIL','AIRL','BC','SQIL','Dagger','BIRL']


    # logs
    exp_id = f"./Log/{env_name}/exp-{num_expert_trajs}/{algorithm}" # task/obj/date structure
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)
    print(f"Logging to directory: {log_folder}")

    os.system(f'cp ./train_imitation.py {log_folder}')
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    print('pid', pid)
    os.makedirs(os.path.join(log_folder, 'plt'))
    os.makedirs(os.path.join(log_folder, 'model'))
    os.makedirs(os.path.join(log_folder, env_name + "_" + algorithm))


    # environment
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    # load expert samples from trained policy
    expert_data = torch.load(f'Expert/{env_name}.pt')

    # tensorboard
    global writer
    writer = tb.SummaryWriter(log_folder + '/' + env_name + "_" + algorithm, flush_secs=1)


    # 开始评估

    for iteration in range(n_itrs):
        print(f"Training with {algorithm}...")
        kl_div,reward = train_algorithm(algorithm, env_name, expert_data, device)

        # 记录到 TensorBoard
        writer.add_scalar( env_name + "/distance", kl_div, iteration)
        writer.add_scalar( env_name + "/reward", reward, iteration)

        logger.record_tabular("iteration", iteration)
        logger.record_tabular("distance", kl_div)
        logger.record_tabular("reward", iteration)
        logger.dump_tabular()

        print(f"[{algorithm}] Iteration {iteration}: KL={kl_div}, Reward={reward}")

    writer.close()

