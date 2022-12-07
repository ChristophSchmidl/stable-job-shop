import gym
from stable_baselines3 import PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from src.callbacks.SaveOnBestTrainingRewardCallback import SaveOnBestTrainingRewardCallback
from src.callbacks.StopTrainingOnMaxEpisodes import StopTrainingOnMaxEpisodes
from src.callbacks.TensorboardCallback import TensorboardCallback
#from src.utils import make_jobshop_env
from src.envs.JobShopEnv.envs.JssEnv import JssEnv
from stable_baselines3.common import results_plotter
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


def plot_vanilla(log_folder, min_len):

    x, y = results_plotter.ts2xy(load_results(log_folder), results_plotter.X_EPISODES)

    x = np.arange(1, len(x) + 1)
    min_len = len(x)


    sns.set_style("whitegrid", {'axes.grid' : True,
                                'axes.edgecolor':'black'

                                })
    fig = plt.figure()
    plt.clf()
    ax = fig.gca()
    colors = ["red", "black", "green", "blue", "purple",  "darkcyan", "brown", "darkblue",]
    labels = ["DQN", "DDQN","Maxmin", "EnsembleDQN", "MaxminDQN"]
    color_patch = []

    for color, label, data in zip(colors, labels, y):
        sns.tsplot(time=range(min_len), data=data, color=color, ci=95)
        color_patch.append(mpatches.Patch(color=color, label=label))
    
    print(min_len)
    plt.xlim([0, min_len])
    plt.xlabel('Training Episodes $(\\times10^6)$', fontsize=22)
    plt.ylabel('Average return', fontsize=22)
    lgd=plt.legend(
    frameon=True, fancybox=True, \
    prop={'weight':'bold', 'size':14}, handles=color_patch, loc="best")
    plt.title('Title', fontsize=14)
    ax = plt.gca()
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    sns.despine()
    plt.tight_layout()
    plt.show()




def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_reward_per_episode(log_folder, title='Learning Curve'):
    """
    plot the reward per episode

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = results_plotter.ts2xy(load_results(log_folder), 'timesteps')

    x = np.arange(1, len(x) + 1)

    #y = moving_average(y, window=50)
    # Truncate x
    #x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()

###############################################################
#                   Create folders and loggers
###############################################################


taillard_instance = "ta41"



log_dir = f"logs/sb3_log/{taillard_instance}"
models_dir = f"models/jss/PPO/{taillard_instance}"
#tensorboard_log = "logs/tensorboard/"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

###############################################################
#                   Create the environment
###############################################################

#env = make_jobshop_env(rank=0, seed=1, instance_name=f"taillard/{taillard_instance}.txt", monitor_log_path=log_dir)
# required before you can step through the environment
#env.reset()


###############################################################
#                   Create the model with callbacks
###############################################################

def create_model(model_name="MaskablePPO", policy="MlpPolicy", env=None, n_env=1, n_steps=20, n_episodes=100, log_dir=None, verbose=1):
    if model_name == "MaskablePPO":
        # Create Callback
        stopTrainingOnMaxEpisodes_callback = StopTrainingOnMaxEpisodes(max_episodes = n_episodes, verbose=verbose)
        tensorboard_callback = TensorboardCallback()
        saveOnBestTrainingReward_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, model_dir=models_dir, verbose=verbose)
        '''
        eval_callback = EvalCallback(env, best_model_save_path='models/jss/PPO/best_model',
                             log_path=log_dir, eval_freq=5,
                             deterministic=False, render=False)
        '''
        # Create the callback list
        callback = CallbackList([stopTrainingOnMaxEpisodes_callback, saveOnBestTrainingReward_callback, tensorboard_callback])

        '''
        clip_param: 0.5653 -> SB3 PPO: clip_range
        entropy_end: 0.00221
        entropy_start: 0.005503
        kl_coeff: 0.116
        kl_target: 0.08849 -> SB3 PPO: target_kl
        layer_size: 264

        lr_end: 0.00009277 -> SB3 PPO: learning_rate     
        lr_start: 0.0008534

        num_sgd_iter: 12 -> SB3 PPO: n_epochs?
        vf_clip_param: 24 -> SB3 PPO: clip_range_vf
        vf_loss_coeff: 0.9991 -> SB3 PPO: vf_coef
        episode_reward_mean: 179.046
        
        '''

        policy_kwargs = dict(activation_fn=th.nn.ReLU,
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
            verbose=verbose, 
            tensorboard_log=log_dir)
        #model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
        return model, callback


###############################################################
#                         Train the model
###############################################################

log_df = pd.DataFrame(columns=['episode', 'timesteps', 'time', 'reward', 'makespan', 'model_id'])
fig = None

TIMESTEPS = np.inf # Dirty hack to make it run forever
for i in range(0, 1):
    env = make_env(rank=0, seed=i, instance_name="taillard/{taillard_instance}.txt", monitor_log_path=log_dir + f"_PPO_{str(i)}_")
    
    model, callback = create_model(
        model_name="MaskablePPO", 
        policy="MlpPolicy", 
        env=env, 
        n_env=1, 
        n_steps=20, 
        n_episodes=25000, 
        log_dir=log_dir, 
        verbose=1)

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True, tb_log_name="PPO", callback=callback, use_masking=True) # TODO: tb_log_name with timestamp, i.e. PPO-{int(time.time())}
    model.save(os.path.join(models_dir, str(TIMESTEPS * i)))

    episode_rewards = env.get_episode_rewards()
    episode_lengths = env.get_episode_lengths()
    episode_times = env.get_episode_times()
    episode_makespans = env.get_episode_makespans()



    log_df = log_df.append(pd.DataFrame({"episode": np.arange(len(episode_rewards)), "timesteps": episode_lengths, "time": episode_times, "reward": episode_rewards, "makespan": episode_makespans, "model_id": i}))

    #print(f"Episode rewards: {env.get_episode_rewards()}")
    #print(f"Episode makespans: {env.get_episode_makespans()}")


    #episodes_df = pd.DataFrame(np.column_stack([]), columns=['episode', 'reward'])
    #log_df = pd.concat([log_df, episode_df])
    #fig = env.render()
    #print(env.get_episode_rewards())
    


env.close()
log_df.reset_index(inplace=True, drop=True) 
log_df.to_csv(log_dir + f"/{taillard_instance}_reward_log.csv")
#df = pd.read_csv(log_dir + '/reward_log.csv', index_col=False)
#print(log_df)
#sns.lineplot(x = "episode", y = "reward", data=log_df)
#plt.show()
