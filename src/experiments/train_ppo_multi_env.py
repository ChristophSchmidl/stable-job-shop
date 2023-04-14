import time
import gym
import numpy as np
import torch as th
import wandb
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.utils import is_masking_supported
from src.utils import evaluate_policy_with_makespan
from src.wrappers import JobShopMonitor
from src.callbacks.TimeLimitCallback import TimeLimitCallback
from src.callbacks.WandbLoggingCallback import WandbLoggingCallback
from stable_baselines3.common.callbacks import CallbackList
from src import config


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
            #env = ActionMasker(env, mask_fn)
            #env = JobShopMonitor(env=env, filename=monitor_log_path) # None means, no log file
            #env = VecMonitor(env, monitor_log_path) # None means, no log file

        if monitor_log_path is not None:
            env = JobShopMonitor(env=env, filename=monitor_log_path) # None means, no log file

        return env

    return _init

def train_agent_multi_env(hyperparam_config, n_envs, input_file, time_limit_in_seconds=5*60):
    # Issues with MaskablePPO and SubprocVecEnv: 
    # https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/49

    instance_name = os.path.split(input_file)[-1].split(sep=".")[0].upper()

    if config.USE_WANDB:
        run = wandb.init(
            project=config.WANDB_PROJECT,
            notes=f"PPO training on instance {instance_name} with tuned hyperparameters. {n_envs} workers with time limit of 30 mins.",
            group="PPO-training-multi-env",
            job_type=f"{instance_name}",
            tags=["ppo", "tuned-hyperparameters", f"{time_limit_in_seconds}-seconds-time-limit", f"{instance_name}"]
        )
    
    wandb.config.update({
        "instance_path": input_file, 
        "time_limit_in_seconds": time_limit_in_seconds
    })

    wandb.config.update({
        "hyperparameters": hyperparam_config
    })


    start_time = time.time()

    # Create a gym environment
    env_name = "jss-v1"
    instance_name = input_file
    n_envs = n_envs

    #seeds = np.random.randint(0, 2**32 - 1, n_envs)
    seed = np.random.randint(0, 2**32 - 1, 1)
    seed_eval = np.random.randint(0, 2**32 - 1, 1)

    # Set up environment
    #env = SubprocVecEnv([make_env(env_id=env_name, rank=i, seed=seed, instance_name=instance_name, permutation_mode=None, permutation_matrix = None, monitor_log_path=None)() for i in range(n_envs)])
    env = make_vec_env(
        make_env(env_id=env_name, rank=0, seed=seed, instance_name=instance_name, permutation_mode=None, permutation_matrix = None, monitor_log_path=None), 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv)

    print(f"Environment: {env} of type {type(env)}")

    print(f"Masking supported on env: {is_masking_supported(env)}")
    
    # Set up evaluation environment
    eval_env = make_env(env_name, rank=0, seed=seed_eval, instance_name=instance_name, permutation_mode=None, permutation_matrix = None, monitor_log_path=None)()
    eval_env = DummyVecEnv([lambda: eval_env])

    print(f"Masking supported on eval_env: {is_masking_supported(eval_env)}")

    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])])
    #policy_kwargs = None

    # configure the PPO agent
    model = MaskablePPO(
            policy='MultiInputPolicy', 
            env=env, 
            learning_rate = hyperparam_config["learning_rate"], # default is 3e-4
            n_steps = hyperparam_config["n_steps"], # default is 2048
            batch_size = hyperparam_config["batch_size"], # default is 64
            n_epochs = hyperparam_config["n_epochs"], # default is 10
            gamma = hyperparam_config["gamma"], # default is 0.99
            gae_lambda = hyperparam_config["gae_lambda"], # default is 0.95
            clip_range = hyperparam_config["clip_range"], # default is 0.2
            clip_range_vf = None, # default is None
            normalize_advantage = True, # default is True
            ent_coef = hyperparam_config["ent_coef"], # default is ent_coef
            vf_coef = hyperparam_config["vf_coef"], # default is 0.5
            max_grad_norm = hyperparam_config["max_grad_norm"], # default is max_grad_norm
            target_kl = None, # default is None
            tensorboard_log = None, # default is None
            #create_eval_env = False, # default is False
            policy_kwargs = policy_kwargs, # default is None
            verbose = 1, # default is 0
            seed = None, # default is None
            device = "auto", # default is "auto"
            _init_setup_model = True # default is True
    )

    if config.USE_WANDB:
        wandb_logging_callback = WandbLoggingCallback(eval_env, n_eval_episodes=5)

    # Set up an evaluation callback
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./models/multi-ppo/",
        log_path="./logs/multi-ppo/",
        eval_freq=100,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Set up timelimit callback
    time_limit_callback = TimeLimitCallback(time_limit_in_seconds)

    if config.USE_WANDB:
        callbacks = CallbackList([wandb_logging_callback, eval_callback, time_limit_callback])
    else:
        callbacks = CallbackList([eval_callback, time_limit_callback])
    

    # Train the PPO agent
    model.learn(total_timesteps=np.inf, callback=callbacks)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Multi-environment training took {elapsed_time:.2f} seconds.")

    # Evaluate the trained agent
    metric_dict = evaluate_policy_with_makespan(model, eval_env, n_eval_episodes=10, deterministic=True)

    if config.USE_WANDB:
        wandb.save('./models/multi-ppo/best_model.zip')

    return metric_dict["mean_reward"], metric_dict["mean_makespan"]


if __name__ == "__main__":
    
    '''
    mean_makespan:  2618
    mean_reward:    103.0101010184735
    '''
    hyperparam_config_first = {
        "clip_range": 0.181648141774528,
        "ent_coef": 0.0033529692788612023,
        "gae_lambda": 0.9981645683766052,
        "gamma": 0.9278778323835192,
        "learning_rate": 0.001080234067815426,
        "max_grad_norm": 7.486785910278103,
        "n_epochs": 7,
        "n_steps": 731,
        "total_timesteps": 81947
    }

    '''
    mean_makespan: 2646
    mean_reward: 97.35353550687432
    '''
    hyperparam_config_second = {
        "clip_range": 0.2515491044924565,
        "ent_coef": 0.006207990430953167,
        "gae_lambda": 0.906079003617699,
        "gamma": 0.9041076240082796,
        "learning_rate": 0.002069479218298502,
        "max_grad_norm": 8.578211744760571,
        "n_epochs": 9,
        "n_steps": 1544,
        "total_timesteps": 69457
    }

    mean_reward, mean_makespan = train_agent_multi_env(hyperparam_config=hyperparam_config_first, n_envs=8, input_file="./data/instances/taillard/ta41.txt", time_limit_in_seconds=5*60)
    print(f"Finished training with mean_reward of {mean_reward} and mean_makespan of {mean_makespan}.")

