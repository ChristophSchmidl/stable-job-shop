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
from src.old_utils import evaluate_policy_with_makespan_single_env, make_env, evaluate_policy_with_makespan



###############################################################
#                           Globals
###############################################################

# TODO: add loop to test all instances with all models

ENV_ID = 'jss-v1'
TAILLARD_INSTANCE = "ta50"
INSTANCE_NAME = f"taillard/{TAILLARD_INSTANCE}.txt"
MODEL_NAME = "best_model_ta50_not_tuned_2500_episodes.zip"
MODEL_DIR = "models/jss/PPO"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
PERMUTATION_MODE = None
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
        
        env = make_env(ENV_ID, 0, 0, instance_name = instance, permutation_mode=None, monitor_log_path=None)()
        env.reset()

        model = MaskablePPOPermutationHandler(model_path=MODEL_PATH, env=env, print_system_info=None)
        mean_reward, std_reward, mean_makespan, std_makespan = evaluate_policy_with_makespan(model, env, n_eval_episodes=1, deterministic=True, use_masking=True)

        rewards_and_makespans["instance_name"].append(instance)
        rewards_and_makespans["rewards"].append(mean_reward)
        rewards_and_makespans["makespans"].append(mean_makespan)

        print(f"Mean reward: {mean_reward}\nStd reward: {std_reward}\nMean makespan: {mean_makespan}\nStd makespan: {std_makespan}")

    return rewards_and_makespans


rewards_and_makespans = evaluate_model_on_instances(EVAL_INSTANCES)

df = pd.DataFrame.from_dict(rewards_and_makespans)
df.to_csv(f"logs/sb3_log/evaluate/evaluate_model_{MODEL_NAME}_on_all_instances.csv")

df['instance_name'] = df['instance_name'].str.replace('taillard/','').str.replace(".txt", '').str.capitalize()
ax = df.set_index("instance_name").plot(kind="bar", figsize=(10,7))
ax.legend(["Reward", "Makespan"], loc="upper right")
ax.set_ylabel("Reward and makespan")
ax.set_xlabel("Instance name", rotation="horizontal")
ax.set_title(f"Makespan and reward for RL {TAILLARD_INSTANCE.capitalize()} policy applied to Taillard instances with 30 jobs and 20 machines")
fig = ax.get_figure()
#plt.draw()
plt.xticks(rotation="horizontal")
fig.savefig(f"plots/2500_episodes/evaluate_policy_{TAILLARD_INSTANCE}_on_30x20_instances.png", dpi=300)
print(df.to_markdown())
plt.show()





