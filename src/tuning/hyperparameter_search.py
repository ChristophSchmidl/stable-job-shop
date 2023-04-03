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
            pass
            env = ActionMasker(env, mask_fn)
            #env = JobShopMonitor(env=env, filename=monitor_log_path) # None means, no log file
            #env = VecMonitor(env, monitor_log_path) # None means, no log file

        if monitor_log_path is not None:
            env = JobShopMonitor(env=env, filename=monitor_log_path) # None means, no log file

        return env

    return _init



def train_agent_single_env(config, input_file):
    start_time = time.time()

    # Create a gym environment
    env_name = "jss-v1"
    instance_name = input_file
    n_envs = 1
    rank = 1
    seed_train = np.random.randint(0, 2**32 - 1, n_envs)
    seed_eval = np.random.randint(0, 2**32 - 1, n_envs)


    env = make_env(env_name, rank=rank, seed=seed_train, instance_name=instance_name, permutation_mode=None, permutation_matrix = None, monitor_log_path=None)()

    # Wrap the environment in a VecEnv
    env = DummyVecEnv([lambda: env])

    # Set up evaluation environment
    eval_env = make_env(env_name, rank=rank, seed=seed_eval, instance_name=instance_name, permutation_mode=None, permutation_matrix = None, monitor_log_path=None)()
    eval_env = DummyVecEnv([lambda: eval_env])

    # configure the PPO agent
    model = MaskablePPO(
            policy='MultiInputPolicy', 
            env=env, 
            learning_rate = config["learning_rate"], # default is 3e-4
            n_steps = config["n_steps"], # default is 2048
            batch_size = 64, # default is 64
            n_epochs = config["n_epochs"], # default is 10
            gamma = config["gamma"], # default is 0.99
            gae_lambda = config["gae_lambda"], # default is 0.95
            clip_range = config["clip_range"], # default is 0.2
            clip_range_vf = None, # default is None
            normalize_advantage = True, # default is True
            ent_coef = config["ent_coef"], # default is ent_coef
            vf_coef = 0.5, # default is vf_coef
            max_grad_norm = config["max_grad_norm"], # default is max_grad_norm
            target_kl = None, # default is None
            tensorboard_log = None, # default is None
            #create_eval_env = False, # default is False
            policy_kwargs = None, # default is None
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

    # Train the PPo agent
    model.learn(total_timesteps=config["total_timesteps"], callback=eval_callback)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Single environment training took {elapsed_time:.2f} seconds.")

    # Evaluate the trained agent
    mean_reward, std_reward, mean_makespan, std_makespan = evaluate_policy_with_makespan(model, eval_env, n_eval_episodes=10, deterministic=True)

    return mean_reward, mean_makespan

def train_agent_multi_env(config, n_envs, input_file):
    # Issues with MaskablePPO and SubprocVecEnv: 
    # https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/49
    start_time = time.time()

    # Create a gym environment
    env_name = "jss-v1"
    instance_name = input_file
    n_envs = n_envs
    rank = 1

    #seeds = np.random.randint(0, 2**32 - 1, n_envs)
    seed = np.random.randint(0, 2**32 - 1, 1)
    seed_eval = np.random.randint(0, 2**32 - 1, n_envs)

    #envs = [make_env(env_id=env_name, rank=rank, seed=int(seed), instance_name=instance_name, permutation_mode=None, permutation_matrix = None, monitor_log_path=None)() for seed in seeds]
    #env = VecEnv(envs)

    #env = make_vec_env(make_env(env_name, rank=rank, seed=seed, instance_name=instance_name, permutation_mode=None, permutation_matrix = None, monitor_log_path=None)(), n_envs, vec_env_cls=DummyVecEnv)
    env = make_vec_env("jss-v1", n_envs, vec_env_cls=DummyVecEnv)
    

    # Set up evaluation environment
    eval_env = make_env(env_name, rank=rank, seed=seed_eval, instance_name=instance_name, permutation_mode=None, permutation_matrix = None, monitor_log_path=None)
    eval_env = DummyVecEnv([lambda: eval_env])

    # configure the PPO agent
    model = MaskablePPO(
            policy='MultiInputPolicy', 
            env=env, 
            learning_rate = config["learning_rate"], # default is 3e-4
            n_steps = config["n_steps"], # default is 2048
            batch_size = 64, # default is 64
            n_epochs = config["n_epochs"], # default is 10
            gamma = config["gamma"], # default is 0.99
            gae_lambda = config["gae_lambda"], # default is 0.95
            clip_range = config["clip_range"], # default is 0.2
            clip_range_vf = None, # default is None
            normalize_advantage = True, # default is True
            ent_coef = config["ent_coef"], # default is ent_coef
            vf_coef = 0.5, # default is vf_coef
            max_grad_norm = config["max_grad_norm"], # default is max_grad_norm
            target_kl = None, # default is None
            tensorboard_log = None, # default is None
            #create_eval_env = False, # default is False
            policy_kwargs = None, # default is None
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

    # Train the PPo agent
    model.learn(total_timesteps=config["total_timesteps"], callback=eval_callback)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Multi-environment training took {elapsed_time:.2f} seconds.")

    # Evaluate the trained agent
    mean_reward, std_reward, mean_makespan, std_makespan = evaluate_policy_with_makespan(model, eval_env, n_eval_episodes=10, deterministic=True)

    return mean_reward, mean_makespan



def run_sweep(tuning_method="bayes", n_runs=20, n_workers=1, input_file="./data/instances/taillard/ta41.txt", project_name="maskable_ppo_hyperparameter_tuning"):

    def sweep_agent_single():
        with wandb.init() as run:
            sweep_config = run.config
            mean_reward, mean_makespan = train_agent_single_env(sweep_config, input_file=input_file)
            wandb.log({"eval/mean_reward": mean_reward, "eval/mean_makespan": mean_makespan})

    def sweep_agent_multi():
        with wandb.init() as run:
            sweep_config = run.config
            mean_reward, mean_makespan = train_agent_multi_env(sweep_config, n_envs=n_workers, input_file=input_file)
            wandb.log({"eval/mean_reward": mean_reward, "eval/mean_makespan": mean_makespan})


    sweep_config = {
        "name": f"maskable_ppo_hyperparameter_tuning_{tuning_method}_{n_runs}-runs_{n_workers}-workers",
        "method": tuning_method,
        "metric": {"goal": "maximize", "name": "eval/mean_reward"},
        "description": f"input_file: {input_file}, n_runs: {n_runs}, n_workers: {n_workers}",
        "parameters": {
            "n_steps": {"min": 64, "max": 2048, "distribution": "int_uniform"},
            "gamma": {"min": 0.9, "max": 0.999, "distribution": "uniform"},
            "learning_rate": {"min": 1e-5, "max": 1e-2, "distribution": "uniform"},
            "ent_coef": {"min": 1e-6, "max": 1e-2, "distribution": "uniform"},
            "clip_range": {"min": 0.1, "max": 0.3, "distribution": "uniform"},
            "n_epochs": {"min": 1, "max": 10, "distribution": "int_uniform"},
            "gae_lambda": {"min": 0.9, "max": 1.0, "distribution": "uniform"},
            "max_grad_norm": {"min": 0.1, "max": 10, "distribution": "uniform"},
            "total_timesteps": {"min": 10_000, "max": 100_000, "distribution": "int_uniform"},
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
            raise NotImplementedError("Multiple envs is not supported yet.")

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