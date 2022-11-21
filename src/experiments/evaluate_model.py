from sb3_contrib.ppo_mask import MaskablePPO
import numpy as np
import gym
import os
import pprint
import pandas as pd
import matplotlib.pyplot as plt
import src.envs.JobShopEnv.envs.JssEnv
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from src.wrappers import JobShopMonitor
#from sb3_contrib.common.maskable.evaluation import evaluate_policy
from src.models import MaskablePPOPermutationHandler
from src.old_utils import evaluate_policy_with_makespan_single_env, make_env




###############################################################
#                           Globals
###############################################################

ENV_ID = 'jss-v1'
INSTANCE_NAME = "taillard/ta41.txt"
MODEL_PATH = "models/jss/PPO/best_model_not_tuned_25k.zip"
PERMUTATION_MODE = True
N_EPISODES = 3000
EVAL_INSTANCES = [f"taillard/ta{i}.txt" for i in range(41,51)]


###############################################################
#                   Create folders and loggers
###############################################################

log_dir = "logs/sb3_log/evaluate"
models_dir = "models/jss/PPO"
#tensorboard_log = "logs/tensorboard/"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)



###############################################################
#                        Create environment
###############################################################

new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

###############################################################
#                   Create the environment
###############################################################

# required before you can step through the environment







#model = MaskablePPO.load("models/jss/PPO/best_model_not_tuned_25k.zip", print_system_info=True)




def evaluate_model_on_instances(instances):
    rewards_and_makespans = {"instance_name": [], "rewards": [], "makespans": []}

    for instance in instances:
        
        env = make_env(ENV_ID, 2, 456, instance_name = instance, permutation_mode=None, monitor_log_path=log_dir + f"_PPO_Permutation_")()
        env.reset()

        model = MaskablePPOPermutationHandler(model_path=MODEL_PATH, env=env, print_system_info=None)
        mean_reward, std_reward, mean_makespan, std_makespan = evaluate_policy_with_makespan_single_env(model, env, n_eval_episodes=1, deterministic=True)

        rewards_and_makespans["instance_name"].append(instance)
        rewards_and_makespans["rewards"].append(mean_reward)
        rewards_and_makespans["makespans"].append(mean_makespan)

        print(f"Mean reward: {mean_reward}\nStd reward: {std_reward}\nMean makespan: {mean_makespan}\nStd makespan: {std_makespan}")

    return rewards_and_makespans


rewards_and_makespans = evaluate_model_on_instances(EVAL_INSTANCES)

df = pd.DataFrame.from_dict(rewards_and_makespans)
df.to_csv("logs/sb3_log/evaluate/evaluate_model_on_instances.csv")

df.set_index("instance_name").plot(kind="bar")
plt.show()





