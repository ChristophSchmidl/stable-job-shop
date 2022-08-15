import gym
from stable_baselines3 import PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from src.callbacks.StopTrainingOnMaxEpisodes import StopTrainingOnMaxEpisodes
from src.callbacks.TensorboardCallback import TensorboardCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.logger import configure

import os
import time
import JSSEnv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


def plot_vanilla(log_folder, min_len):

    x, y = results_plotter.ts2xy(load_results(log_folder), 'timesteps')

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

log_dir = "logs/sb3_log/"
models_dir = "models/jss/PPO"
#tensorboard_log = "logs/tensorboard/"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

###############################################################
#                   Create the environment
###############################################################

# See: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/docs/modules/ppo_mask.rst
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_legal_actions()

# Create the environment
env = gym.make('jss-v1', env_config={'instance_path': './data/instances/taillard/ta41.txt'})
env = ActionMasker(env, mask_fn)
# Info on monitor.csv
# ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
env = Monitor(env=env, filename=log_dir)

# required before you can step through the environment
env.reset()


###############################################################
#                   Create the model with callbacks
###############################################################


# Create Callback
stopTrainingOnMaxEpisodes_callback = StopTrainingOnMaxEpisodes(max_episodes = 20, verbose=1)
tensorboard_callback = TensorboardCallback()
eval_callback = EvalCallback(env, best_model_save_path='models/jss/PPO/best_model',
                             log_path=log_dir, eval_freq=5,
                             deterministic=False, render=False)


# Create the callback list
callback = CallbackList([stopTrainingOnMaxEpisodes_callback, tensorboard_callback])

model = MaskablePPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir)
#model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
model.set_logger(new_logger)


###############################################################
#                         Train the model
###############################################################


fig = None

TIMESTEPS = np.inf # Dirty hack to make it run forever
for i in range(1, 2):
    # Pass reset_num_timesteps=False to continue the training curve in tensorboard
    # By default, it will create a new curve
    # If you specify different tb_log_name in subsequent runs, you will have split graphs
    # See: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", callback=callback) # TODO: tb_log_name with timestamp, i.e. PPO-{int(time.time())}
    model.save(os.path.join(models_dir, str(TIMESTEPS * i)))
    print(f"Last time step: {env.last_time_step}")

    # Grab the monitor.csv file and put it into a pandas dataframe?
    #df = pd.read_csv(os.path.join(log_dir, 'monitor.csv'))

    fig = env.render()

env.close()

#fig.show()
#results_plotter.plot_results([log_dir], None, results_plotter.X_EPISODES, "JOB-SHOP")
#plt.show()
#plot_vanilla(log_dir)

