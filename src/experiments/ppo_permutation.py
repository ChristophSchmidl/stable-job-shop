import os
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from src.models import MaskablePPOPermutationHandler
from src.envs.JobShopEnv.envs.JssEnv import JssEnv
from src.io.jobshoploader import JobShopLoader
from src.old_utils import make_env, evaluate_policy_with_makespan
import pprint


###############################################################
#                           Globals
###############################################################

ENV_ID = 'jss-v1'
INSTANCE_NAME = "taillard/ta41.txt"
MODEL_PATH = "models/jss/PPO/best_model_not_tuned_25k.zip"
PERMUTATION_MODE = True
N_EPISODES = 3000


###############################################################
#                   Create folders and loggers
###############################################################

log_dir = "logs/sb3_log/ppo_permutation"
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
env = DummyVecEnv([make_env(ENV_ID, 2, 456, instance_name = "taillard/ta41.txt", permutation_mode=PERMUTATION_MODE, monitor_log_path=log_dir + f"_PPO_Permutation_")])
# required before you can step through the environment
env.reset()

###############################################################
#                   Create the RL agent
###############################################################

job_count, machine_count, list_of_jobs = JobShopLoader.load_jssp_instance_as_list(f"./data/instances/{INSTANCE_NAME}")
#pprint.pprint(f"Original list of jobs: {list_of_jobs}")
#print("\n")

# Permutate list of jobs
#permuted_list_of_jobs, perm_matrix, perm_indices = permute_instance(list_of_jobs)
#print.pprint(f"Permuted list of jobs: {permuted_list_of_jobs}")
#print("\n")

model = MaskablePPOPermutationHandler(model_path=MODEL_PATH, env=env, print_system_info=None)


###############################################################
#                  Test the permutation mode
###############################################################

mean_reward, std_reward, mean_makespan, std_makespan = evaluate_policy_with_makespan(model, env, n_eval_episodes=1, deterministic=True, use_masking=True)

print(f"Mean reward: {mean_reward}\nStd reward: {std_reward}\nMean makespan: {mean_makespan}\nStd makespan: {std_makespan}")