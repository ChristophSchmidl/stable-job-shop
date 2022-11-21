import os
import numpy as np
from typing import Dict, Any
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from src.models import MaskablePPOPermutationHandler
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from src.envs.JobShopEnv.envs.JssEnv import JssEnv
from src.io.jobshoploader import JobShopLoader
from src.old_utils import make_env, evaluate_policy_with_makespan
from src.utils.experience_collector import ExperienceCollector
import gc
import pprint


###############################################################
#                           Globals
###############################################################

ENV_ID = 'jss-v1'
INSTANCE_NAME = "taillard/ta41.txt"
MODEL_PATH = "models/jss/PPO/best_model_not_tuned_25k.zip"
N_EPISODES = 1000
SAVE_EVERY_N_EPISODES = 100
#N_SWAPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
N_SWAPS = [8, 9, 10, 11, 12, 13, 14, 15]



###############################################################
#                   Create folders and loggers
###############################################################

log_dir = "logs/sb3_log/ppo_permutation"
models_dir = "models/jss/PPO"
data_dir = "data/experiences"
#tensorboard_log = "logs/tensorboard/"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)


###############################################################
#                        Create environment
###############################################################

new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])


###############################################################
#              Collect normal experiences (one episode is enough because deterministic)
###############################################################
'''
collector = ExperienceCollector(instance_name="taillard/ta41.txt", model_path="models/jss/PPO/best_model_not_tuned_25k.zip", n_episodes=N_EPISODES, save_every_n_episodes=SAVE_EVERY_N_EPISODES, permutation_mode=None, data_dir="data/experiences")
collector.start()
del collector
gc.collect()
'''

###############################################################
#              Collect random permuted experiences
###############################################################
'''
collector = ExperienceCollector(instance_name="taillard/ta41.txt", model_path="models/jss/PPO/best_model_not_tuned_25k.zip", n_episodes=N_EPISODES, save_every_n_episodes=SAVE_EVERY_N_EPISODES, permutation_mode="random", data_dir="data/experiences")
collector.start()
del collector
gc.collect()
'''

###############################################################
#              Collect transposed experiences with n_swap
###############################################################

for swap in N_SWAPS:
    collector = ExperienceCollector(instance_name="taillard/ta41.txt", model_path="models/jss/PPO/best_model_not_tuned_25k.zip", n_episodes=N_EPISODES, save_every_n_episodes=SAVE_EVERY_N_EPISODES, permutation_mode=f"transpose={swap}", data_dir="data/experiences")
    collector.start()
    del collector
    gc.collect()