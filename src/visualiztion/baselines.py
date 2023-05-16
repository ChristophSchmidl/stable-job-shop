import os
import gym
import numpy as np
from sb3_contrib import MaskablePPO
from src.utils import evaluate_policy_with_makespan
from src.wrappers import JobShopMonitor
from stable_baselines3.common.vec_env import DummyVecEnv
import time


'''
1. RL: Load best_model.zip for Ta41 - Ta50 and calculate their makespans based on eval_env
2. Dispatching rules: Apply dispatching rules to Ta41 - Ta50
3. Constraint-programming: Load the dataframes? From wandb?
'''

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

def evaluate_8_hour_ta41_rl():
    # Set up evaluation environment
    instance_path = f"./data/instances/taillard/ta41.txt"
    best_model_path = f"./models/multi-ppo-tuned/ta41/best_model_8_hours.zip"
    mean_reward, mean_makespan, elapsed_time = evaluate_rl_model(best_model_path, eval_env)
    print(f"Mean makespan: {mean_makespan}")

def evaluate_rl_model(model_path="./models/trained_tuned_30_mins/ta41/best_model.zip", eval_instance_path="./data/instances/taillard/ta41.txt", eval_permutation_mode=None, n_eval_episodes=100):
    '''
    returns makespan, inference_time
    '''
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"The file '{model_path}' does not exist.")
    
    if not os.path.isfile(eval_instance_path):
        raise FileNotFoundError(f"The file '{eval_instance_path}' does not exist.")

    # Create evaluation environment
    seed_eval = np.random.randint(0, 2**32 - 1, 1)
    eval_env = make_env("jss-v1", rank=0, seed=seed_eval, instance_name=eval_instance_path, permutation_mode=eval_permutation_mode, permutation_matrix = None, monitor_log_path=None)()
    eval_env = DummyVecEnv([lambda: eval_env])

    # Load model
    best_model = MaskablePPO.load(model_path, eval_env)

    start_time = time.perf_counter()
    metric_dict = evaluate_policy_with_makespan(best_model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"The model {model_path} took {elapsed_time:.6f} seconds to execute for env {eval_instance_path}. Mean reward: {metric_dict['mean_reward'] }, Mean makespan: {metric_dict['mean_makespan']} ")

    return metric_dict["mean_reward"], metric_dict["mean_makespan"], elapsed_time
    
def get_baseline_rl_ta41_applied_to_others_makespans():
    instance_names = [f"ta{instance_index}" for instance_index in range(41,51)]
    instance_paths = [f"./data/instances/taillard/ta{instance_index}.txt" for instance_index in range(41, 51)]

    makespans_and_timings = {}

    for instance_name, instance_path in zip(instance_names, instance_paths):
        best_model_path = f"./models/trained_tuned_30_mins/ta41/best_model.zip"
        mean_reward, mean_makespan, elapsed_time = evaluate_rl_model(model_path=best_model_path, eval_instance_path=instance_path, eval_permutation_mode=None)

        makespans_and_timings.update({f"{instance_name}-makespan": mean_makespan, f"{instance_name}-timing": elapsed_time})

    print(makespans_and_timings)
    return makespans_and_timings


def get_baseline_rl_makespans():
    instance_names = [f"ta{instance_index}" for instance_index in range(41,51)]
    instance_paths = [f"./data/instances/taillard/ta{instance_index}.txt" for instance_index in range(41, 51)]

    makespans_and_timings = {}

    for instance_name, instance_path in zip(instance_names, instance_paths):
        # Set up evaluation environment
        seed_eval = np.random.randint(0, 2**32 - 1, 1)
        eval_env = make_env("jss-v1", rank=0, seed=seed_eval, instance_name=instance_path, permutation_mode=None, permutation_matrix = None, monitor_log_path=None)()
        eval_env = DummyVecEnv([lambda: eval_env])

        best_model_path = f"./models/trained_tuned_30_mins/{instance_name}/best_model.zip"
        mean_reward, mean_makespan, elapsed_time = evaluate_rl_model(best_model_path, eval_env)

        makespans_and_timings.update({f"{instance_name}-makespan": mean_makespan, f"{instance_name}-timing": elapsed_time})

    print(makespans_and_timings)
    return makespans_and_timings





