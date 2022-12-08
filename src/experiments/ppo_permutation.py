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
TAILLARD_INSTANCE = "ta41"
INSTANCE_NAME = f"taillard/{TAILLARD_INSTANCE}.txt"
PERMUTATION_MODE = None
N_EPISODES = 100
MODEL_DIR = f"models/jss/PPO"
#MODEL_FILENAME = f"best_model_{TAILLARD_INSTANCE}_not_tuned_{N_EPISODES}_episodes.zip"
#MODEL_FILE = os.path.join(MODEL_DIR, MODEL_FILENAME)

#MODEL_PATH = f"models/jss/PPO/best_model_{TAILLARD_INSTANCE}_not_tuned_{N_EPISODES}_episodes.zip"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model_ta41_not_tuned_25k.zip")

###############################################################
#                   Create folders and loggers
###############################################################

log_dir = "logs/sb3_log/ppo_permutation"
models_dir = MODEL_DIR
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
env = make_env(ENV_ID, 2, 456, instance_name = f"taillard/{TAILLARD_INSTANCE}.txt", permutation_mode=PERMUTATION_MODE, monitor_log_path=None)()
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