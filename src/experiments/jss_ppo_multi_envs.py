from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from src.callbacks.SaveOnBestTrainingRewardCallback import SaveOnBestTrainingRewardCallback
from src.callbacks.StopTrainingOnMaxEpisodes import StopTrainingOnMaxEpisodes
from src.callbacks.TensorboardCallback import TensorboardCallback
from src.utils import make_env
from src.envs.JobShopEnv.envs.JssEnv import JssEnv
import gym
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch as th

###############################################################
#                           Globals
###############################################################

env_id = 'jss-v1'
num_process = 4
num_experiments = 3
verbose = 1
n_episodes = 3000

###############################################################
#                   Create folders and loggers
###############################################################

log_dir = "logs/sb3_log/PPO_MULTI_ENVS"
models_dir = "models/jss/PPO_MULTI_ENVS"
#tensorboard_log = "logs/tensorboard/"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])


###############################################################
#                         Create envs
###############################################################
eval_env = DummyVecEnv([make_env(env_id, 0, 0, 'taillard/ta41.txt', shuffle_instance=False, monitor_log_path=log_dir + f"_PPO_MULTI_")])
train_env = SubprocVecEnv([make_env(env_id, rank=0, seed=1, instance_name="taillard/ta41.txt", shuffle_instance=True, monitor_log_path=log_dir + f"_PPO_{str(i)}_") for i in range(num_process)], start_method='fork')


###############################################################
#                         Train the model
###############################################################

log_df = pd.DataFrame(columns=['episode', 'timesteps', 'time', 'reward', 'makespan', 'model_id'])
fig = None
TIMESTEPS = np.inf # Dirty hack to make it run forever


for experiment in range(num_experiments):
    # it is recommended to run several experiments due to variability in results
    train_env.reset()

    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])])
    #policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 256, 256])

    model = MaskablePPO(
    policy='MultiInputPolicy', # alias of MaskableMultiInputActorCriticPolicy
    env=train_env, 
    policy_kwargs=policy_kwargs,
    verbose=verbose)

    stopTrainingOnMaxEpisodes_callback = StopTrainingOnMaxEpisodes(max_episodes = n_episodes, verbose=verbose)
    #tensorboard_callback = TensorboardCallback()
    #saveOnBestTrainingReward_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, model_dir=models_dir, verbose=verbose)
    callback = CallbackList([stopTrainingOnMaxEpisodes_callback ])

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True, tb_log_name="PPO_MULTI", callback=callback, use_masking=True) # TODO: tb_log_name with timestamp, i.e. PPO-{int(time.time())}
    model.save(os.path.join(models_dir, str(TIMESTEPS * experiment)))

    # Do I have to evaluate in between with 'evaluate_policy' applied to the eval_env?

    episode_rewards = train_env.get_episode_rewards()
    episode_lengths = train_env.get_episode_lengths()
    episode_times = train_env.get_episode_times()
    episode_makespans = train_env.get_episode_makespans()

    log_df = log_df.append(pd.DataFrame({"episode": np.arange(len(episode_rewards)), "timesteps": episode_lengths, "time": episode_times, "reward": episode_rewards, "makespan": episode_makespans, "model_id": i}))

    #mean_reward, _  = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
    #rewards.append(mean_reward)
# Important: when using subprocess, don't forget to close them
# otherwise, you may have memory issues when running a lot of experiments
train_env.close()

log_df.reset_index(inplace=True, drop=True) 
log_df.to_csv(log_dir + '/reward_log.csv')
#df = pd.read_csv(log_dir + '/reward_log.csv', index_col=False)
print(log_df)
sns.lineplot(x = "episode", y = "reward", data=log_df)
plt.show()

