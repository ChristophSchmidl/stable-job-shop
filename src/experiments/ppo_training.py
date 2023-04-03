import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch as T
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
from src.models import MaskablePPOPermutationHandler
from src.envs.JobShopEnv.envs.JssEnv import JssEnv
from src.io.jobshoploader import JobShopLoader
from src.callbacks.SaveOnBestTrainingRewardCallback import SaveOnBestTrainingRewardCallback
from src.callbacks.StopTrainingOnMaxEpisodes import StopTrainingOnMaxEpisodes
import pprint

###############################################################
#                           Globals
###############################################################
print_device_info()

DEVICE = get_device('gpu')
VERBOSE = True
ENV_ID = 'jss-v1'
TAILLARD_INSTANCE = "ta50"
INSTANCE_NAME = f"taillard/{TAILLARD_INSTANCE}.txt"
PERMUTATION_MODE = None
N_EPISODES = 2500
MODEL_DIR = f"models/jss/PPO"
MODEL_FILENAME = f"best_model_{TAILLARD_INSTANCE}_not_tuned_{N_EPISODES}_episodes.zip"
MODEL_FILE = os.path.join(MODEL_DIR, MODEL_FILENAME)
LOG_DIR = f"logs/sb3_log/ppo_training_{TAILLARD_INSTANCE}"
TENSORBOARD_LOG_DIR = "logs/tensorboard/"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# I need this for the creation of the monitor.csv? Nope, just the MonitorWrapper
#new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
###############################################################
#                   Create the environment
###############################################################

#new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

env = DummyVecEnv([make_env(ENV_ID, 2, 456, instance_name = f"taillard/{TAILLARD_INSTANCE}.txt", permutation_mode=PERMUTATION_MODE, monitor_log_path=LOG_DIR)])
# required before you can step through the environment
env.reset()

###############################################################
#                   Create callbacks
###############################################################

stopTrainingOnMaxEpisodes_callback = StopTrainingOnMaxEpisodes(max_episodes = N_EPISODES, verbose=VERBOSE)
#tensorboard_callback = TensorboardCallback()
saveOnBestTrainingReward_callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=LOG_DIR, model_dir=MODEL_DIR, model_filename=MODEL_FILENAME, verbose=VERBOSE)
callbacks = CallbackList([stopTrainingOnMaxEpisodes_callback, saveOnBestTrainingReward_callback])

###############################################################
#                   Create agent
###############################################################

policy_kwargs = dict(activation_fn=T.nn.ReLU,
                    net_arch=[264])

model = MaskablePPO(
            policy='MultiInputPolicy', 
            env=env, 
            #clip_range=0.5653,
            #target_kl=0.08849,
            #learning_rate=0.0008534,
            #n_epochs=12,
            #clip_range_vf=24,
            #vf_coef=0.9991,
            #policy_kwargs=policy_kwargs,
            device=DEVICE,
            verbose=VERBOSE, 
            tensorboard_log=TENSORBOARD_LOG_DIR)

#model.set_logger(new_logger)

log_df = pd.DataFrame(columns=['episode', 'timesteps', 'time', 'reward', 'makespan', 'model_id'])
fig = None

model.learn(total_timesteps=np.inf, reset_num_timesteps=True, callback=callbacks, use_masking=True) # TODO: tb_log_name with timestamp, i.e. PPO-{int(time.time())}
#model.save(os.path.join(models_dir, str(TIMESTEPS * i)))
# I have to use env_method and [0] at
episode_rewards = env.env_method("get_episode_rewards")[0]
episode_lengths = env.env_method("get_episode_lengths")[0]
episode_times = env.env_method("get_episode_times")[0]
episode_makespans = env.env_method("get_episode_makespans")[0]

log_df = log_df.append(pd.DataFrame({"episode": np.arange(len(episode_rewards)), "timesteps": episode_lengths, "time": episode_times, "reward": episode_rewards, "makespan": episode_makespans, "model_id": 0}))

env.close()
log_df.reset_index(inplace=True, drop=True) 
log_df.to_csv(LOG_DIR + f"/{TAILLARD_INSTANCE}_reward_log.csv")

#df = pd.read_csv(log_dir + '/reward_log.csv', index_col=False)
print(log_df)
#sns.lineplot(x = "episode", y = "reward", data=log_df)
#plt.show()