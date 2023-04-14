import os
import random
import pprint
import time
import numpy as np
import gym
import wandb
import torch
from wandb.integration.sb3 import WandbCallback
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib.common.wrappers import ActionMasker
from src.wrappers import JobShopMonitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from src.envs.JobShopEnv.envs.JssEnv import JssEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.utils import is_masking_supported
from src.callbacks.StopTrainingOnMaxEpisodes import StopTrainingOnMaxEpisodes
from stable_baselines3.common.callbacks import CallbackList
from src.utils import evaluate_policy_with_makespan


# See: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/docs/modules/ppo_mask.rst
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_legal_actions()

def make_env(env_id, rank=0, seed=0, instance_name="./data/instances/taillard/ta01.txt", permutation_mode=None, permutation_matrix = None, monitor_log_path=None):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env_config = {"instance_path": instance_name, "permutation_mode": permutation_mode, "permutation_matrix": permutation_matrix}

        env = gym.make(env_id, env_config=env_config)
        # Important: use a different seed for each environment
        if rank is not None and seed is not None:
            env.seed(seed + rank)
        
        if env_id == "jss-v1":
            env = ActionMasker(env, mask_fn)
            #env = JobShopMonitor(env=env, filename=monitor_log_path) # None means, no log file
            #env = VecMonitor(env, monitor_log_path) # None means, no log file

        if monitor_log_path is not None:
            env = JobShopMonitor(env=env, filename=monitor_log_path) # None means, no log file

        return env

    return _init

def train_agent(config, n_envs, input_file, max_episodes=30):
    start_time = time.time()

    assert n_envs >= 1

    # Create a gym environment
    env_name = "jss-v1"
    instance_name = input_file
    n_envs = n_envs
    rank = 1

    seeds = np.random.randint(0, 2**32 - 1, n_envs)
    seed_eval = np.random.randint(0, 2**32 - 1, n_envs)

    if n_envs == 1:
        env = make_env(
            env_name, 
            rank=rank, 
            seed=seeds, 
            instance_name=instance_name, 
            permutation_mode=None, 
            permutation_matrix = None, 
            monitor_log_path="./logs/monitor_logs/sweeps/test_")()
        #env = DummyVecEnv([lambda: env])
    elif n_envs > 1:
        # Issues with MaskablePPO and SubprocVecEnv: 
        # https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/49
        env = make_vec_env(
            make_env(env_name, rank=rank, seed=seed_eval, instance_name=instance_name, permutation_mode=None, permutation_matrix = None, monitor_log_path=None)
            , n_envs, vec_env_cls=SubprocVecEnv)

    # Set up evaluation environment
    eval_env = make_env(env_name, rank=rank, seed=seed_eval, instance_name=instance_name, permutation_mode=None, permutation_matrix = None, monitor_log_path=None)()
    eval_env = DummyVecEnv([lambda: eval_env])

    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[319, 319], vf=[319, 319])])

    # configure the PPO agent
    model = MaskablePPO(
            policy='MultiInputPolicy', 
            env=env, 
            learning_rate = config["learning_rate"], # default is 3e-4
            n_steps = config["n_steps"], # default is 2048
            batch_size = config["batch_size"], # default is 64
            n_epochs = config["n_epochs"], # default is 10
            gamma = config["gamma"], # default is 0.99
            gae_lambda = config["gae_lambda"], # default is 0.95
            clip_range = config["clip_range"], # default is 0.2
            clip_range_vf = None, # default is None
            normalize_advantage = True, # default is True
            ent_coef = config["ent_coef"], # default is ent_coef
            vf_coef = config["vf_coef"], # default is 0.5
            max_grad_norm = config["max_grad_norm"], # default is max_grad_norm
            target_kl = None, # default is None
            tensorboard_log = None, # default is None
            #create_eval_env = False, # default is False
            policy_kwargs = policy_kwargs, # default is None
            verbose = 1, # default is 0
            seed = None, # default is None
            device = "auto", # default is "auto"
            _init_setup_model = True # default is True
    )

    # Set up an evaluation callback
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./models/sweeps/",
        log_path="./logs/sweeps/",
        eval_freq=500,
        deterministic=True,
        render=False
    )

    stopTrainingOnMaxEpisodes_callback = StopTrainingOnMaxEpisodes(max_episodes = max_episodes, verbose=1)

    callbacks = CallbackList([stopTrainingOnMaxEpisodes_callback, eval_callback])

    # Train the PPO agent
    model.learn(total_timesteps=np.inf, callback=callbacks)

    # Get the episode data
    episode_rewards = env.get_episode_rewards()
    episode_lengths = env.get_episode_lengths()
    episode_times = env.get_episode_times()
    episode_makespans = env.get_episode_makespans()

    episode_rewards_min = np.min(episode_rewards)
    episode_rewards_max = np.max(episode_rewards)
    episode_rewards_mean = np.mean(episode_rewards)

    episode_lengths_min = np.min(episode_lengths)
    episode_lengths_max = np.max(episode_lengths)
    episode_lengths_mean = np.mean(episode_rewards)

    episode_times_min = np.min(episode_times)
    episode_times_max = np.max(episode_times)
    episode_times_mean = np.mean(episode_times)

    episode_makespans_min = np.min(episode_makespans)
    episode_makespans_max = np.max(episode_makespans)
    episode_makespans_mean = np.mean(episode_makespans)

    # Log the episode data
    wandb.log({
        "eval/episode_rewards": episode_rewards,
        "eval/episode_rewards_min": episode_rewards_min,
        "eval/episode_rewards_max": episode_rewards_max,
        "eval/episode_rewards_mean": episode_rewards_mean
    })

    wandb.log({
        "eval/episode_lengths": episode_lengths,
        "eval/episode_lengths_min": episode_lengths_min,
        "eval/episode_lengths_max": episode_lengths_max,
        "eval/episode_lengths_mean": episode_lengths_mean
    })

    wandb.log({
        "eval/episode_times": episode_times,
        "eval/episode_times_min": episode_times_min,
        "eval/episode_times_max": episode_times_max,
        "eval/episode_times_mean": episode_times_mean
    })

    wandb.log({
        "eval/episode_makespans": episode_makespans,
        "eval/episode_makespans_min": episode_makespans_min,
        "eval/episode_makespans_max": episode_makespans_max,
        "eval/episode_makespans_mean": episode_makespans_mean
    })

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    if n_envs == 1:
        print(f"Single environment training took {elapsed_time:.2f} seconds.")
    elif n_envs > 1:
        print(f"Multi-environment training took {elapsed_time:.2f} seconds.")

    # Evaluate the trained agent
    metric_dict = evaluate_policy_with_makespan(model, eval_env, n_eval_episodes=10, deterministic=True)

    return metric_dict["mean_reward"], metric_dict["mean_makespan"]

def run_sweep(tuning_method="bayes", n_runs=20, n_workers=1, max_episodes=30, input_file="./data/instances/taillard/ta41.txt", project_name="maskable_ppo_hyperparameter_tuning"):

    def sweep_agent_single():
        with wandb.init() as run:
            sweep_config = run.config
            
            mean_reward, mean_makespan = train_agent(sweep_config, 1, input_file, max_episodes=max_episodes)
            # eval/mean_reward and eval/mean_makespan seem to give the mean over all eval env episodes
            #wandb.log({"eval/mean_reward": mean_reward, "eval/mean_makespan": mean_makespan})

    def sweep_agent_multi():
        with wandb.init() as run:
            sweep_config = run.config
            mean_reward, mean_makespan = train_agent(sweep_config, n_workers, input_file, max_episodes=max_episodes)
            #wandb.log({"eval/mean_reward": mean_reward, "eval/mean_makespan": mean_makespan})

    sweep_config = {
        "name": f"maskable_ppo_hyperparameter_tuning_{tuning_method}_{n_runs}-runs_{n_workers}-workers",
        "method": tuning_method,
        "metric": {"goal": "maximize", "name": "eval/mean_reward"},
        "description": f"input_file: {input_file}, n_runs: {n_runs}, n_workers: {n_workers}",
        "parameters": {
            "batch_size": {"min": 64, "max": 512, "distribution": "int_uniform"},
            "n_steps": {"min": 64, "max": 2048, "distribution": "int_uniform"},
            "gamma": {"min": 0.9, "max": 0.999, "distribution": "uniform"},
            "learning_rate": {"min": 1e-5, "max": 1e-2, "distribution": "uniform"},
            "ent_coef": {"min": 0.0, "max": 0.01, "distribution": "uniform"},
            "vf_coef": {"min": 0.5, "max": 1.0, "distribution": "uniform"},
            "clip_range": {"min": 0.1, "max": 0.6, "distribution": "uniform"},
            "n_epochs": {"min": 3, "max": 30, "distribution": "int_uniform"},
            "gae_lambda": {"min": 0.9, "max": 1.0, "distribution": "uniform"},
            "max_grad_norm": {"min": 0.1, "max": 10, "distribution": "uniform"},
            #"total_timesteps": {"min": 10_000, "max": 100_000, "distribution": "int_uniform"},
            #"total_episodes": {"min":30, "max": 30, "distribution": "int_uniform"}
        },
    }


    if tuning_method == "bayes":
        if n_workers == 1:
            # single
            single_sweep_id = wandb.sweep(sweep_config, project=project_name)
            wandb.agent(single_sweep_id, function=sweep_agent_single, count=n_runs)
        elif n_workers > 1:
            raise ValueError("Bayes method does not support multiple workers. n_workers has to be 1.")

    elif tuning_method == "random":
        if n_workers == 1:
            single_sweep_id = wandb.sweep(sweep_config, project=project_name)
            wandb.agent(single_sweep_id, function=sweep_agent_single, count=n_runs)
        elif n_workers > 1:
            multi_sweep_id = wandb.sweep(sweep_config, project=project_name)
            wandb.agent(multi_sweep_id, function=sweep_agent_multi, count=n_runs)

    elif tuning_method == "grid":
        if n_workers == 1:
            single_sweep_id = wandb.sweep(sweep_config, project=project_name)
            wandb.agent(single_sweep_id, function=sweep_agent_single, count=n_runs)
        elif n_workers > 1:
            raise NotImplementedError("Multiple envs is not supported yet.")














'''

def train():
    with wandb.init() as run:
        config = wandb.config

        # Create the environment
        env = make_jobshop_env(rank=0, seed=1, instance_name="taillard/ta41.txt", monitor_log_path=log_dir)
        # required before you can step through the environment
        env.reset()

        # Create the model with callbacks
        model, callback = create_model(
            model_name="MaskablePPO", 
            policy="MlpPolicy", 
            env=env, 
            n_env=1, 
            n_steps=20, 
            n_episodes=25000, 
            log_dir=log_dir, 
            verbose=1)

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True, tb_log_name="PPO", callback=callback, use_masking=True)

    model.learn(total_timesteps=episodes, reset_num_timesteps=True, tb_log_name="PPO", callback=WandbCallback(), use_masking=True) # TODO: tb_log_name with timestamp, i.e. PPO-{int(time.time())}
    model.save(os.path.join(wandb.run.dir, "models", str(episodes * i)))

    # Get the episode data
    episode_rewards = env.get_episode_rewards()
    episode_lengths = env.get_episode_lengths()
    episode_times = env.get_episode_times()
    episode_makespans = env.get_episode_makespans()

    # Log the episode data
    wandb.log({
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_times": episode_times,
        "episode_makespans": episode_makespans
    })

    # Log the model
    wandb.save(os.path.join(wandb.run.dir, "models", str(episodes * i)))

    # Log the environment
    wandb.save(os.path.join(wandb.run.dir, "env", str(episodes * i)))




sweep_config = {
    "name": "jss-ppo-sweep",
    "method": "bayes",
    "metric": {
        "name": "episode_rewards",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {
            "values": [0.0008, 0.00009]
        },
        "clip_range": {
            "values": [0.1, 0.6]
        },
        "clip_range_vf": {
            "values": [0.1, 24]
        },
        "gamma": {
            "values": [0.99, 0.999]
        },
        "vf_coef": {
            "values": [0.5, 0.9999]
        },
        "ent_coef": {
            "values": [0.002, 0.006]
        },
        "target_kl": {
            "values": [0.088, 0.116]
        },
        "optimizer": {
            "values": ["adam", "sgd"]
        },
    }
}

wandb.init(config=sweep_config)
config = wandb.config



sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train)



env = make_jobshop_env(rank=0, seed=1, instance_name="taillard/ta41.txt", monitor_log_path=run.dir)


policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=[256, 256]
    lr_schedule=0.00025,
    optimizer_kwargs=dict(eps=1e-5),
    )
'''