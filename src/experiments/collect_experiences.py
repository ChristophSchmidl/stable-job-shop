import os
from typing import Dict, Any
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from src.models import MaskablePPOPermutationHandler
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from src.envs.JobShopEnv.envs.JssEnv import JssEnv
from src.io.jobshoploader import JobShopLoader
from src.old_utils import make_env, evaluate_policy_with_makespan
from src.callbacks.CollectExperiencesCallback import CollectExperiencesCallback
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


def collect_experiences(_locals: Dict[str, Any], _globals: Dict[str, Any]):
    # TODO: 1. Refactoring: Create own class for collecting experiences. ExperienceCollector?
    #       1a. Using this function in combination with evaluate_policy does not seem to make sense.
    # TODO: 2. What do we want to save?
    # TODO: 3. How do we want to save it? (TXT, CSV, JSON, ...)
    # TODO: 4. How do we want to save it? (One file per episode, one file per run, ...)
    # TODO: 5. Where do want to save it? (In the same folder as the model, in a separate folder, ...)

    #pprint.pprint(f"Printing _locals: {_locals}")
    #pprint.pprint(f"Printing _locals: {_globals}")
    # What we want to collect: states, actions, rewards, dones, infos
    state = _locals["observations"]["real_obs"] 
    boolean_action_mask = _locals["action_masks"] # This is a boolean mask
    action = _locals["actions"] # This is the action that was actually taken. Unwrapping?
    reward = _locals["reward"]
    done = _locals["done"]
    makespan = _locals["info"]["makespan"]

    print(f"Printing state: {state}")
    pprint.pprint(f"Printing action: {action[0]}")


model = MaskablePPOPermutationHandler(model_path=MODEL_PATH, env=env, print_system_info=None)


#collect_experiences_callback = CollectExperiencesCallback(verbose=1)

#model.collect_rollouts(env=env, callback=base_callback, rollout_buffer=rollout_buffer, n_rollout_steps=1000, use_masking=True)

###############################################################
#                  Test the permutation mode
###############################################################


mean_reward, std_reward, mean_makespan, std_makespan = evaluate_policy_with_makespan(
    model=model, 
    env=env, 
    n_eval_episodes=1, 
    deterministic=True, 
    callback=collect_experiences,
    use_masking=True
    )


print(f"Mean reward: {mean_reward}\nStd reward: {std_reward}\nMean makespan: {mean_makespan}\nStd makespan: {std_makespan}")