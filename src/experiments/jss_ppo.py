import gym
from stable_baselines3 import PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from src.callbacks.StopTrainingOnMaxEpisodes import StopTrainingOnMaxEpisodes
from stable_baselines3.common import results_plotter
import os
import time
import JSSEnv
import numpy as np
import matplotlib.pyplot as plt

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

log_dir = f"logs/sb3_log/"
models_dir = "models/jss/PPO"

# set up logger

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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
env = Monitor(env, log_dir)

# required before you can step through the environment
env.reset()


###############################################################
#                   Create the model with callbacks
###############################################################


# Create Callback
callback = StopTrainingOnMaxEpisodes(max_episodes = 100, verbose=1)


model = MaskablePPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir)
#model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)



###############################################################
#                         Train the model
###############################################################


fig = None

TIMESTEPS = np.inf # Dirty hack to make it run forever
for i in range(1, 5):
    # Pass reset_num_timesteps=False to continue the training curve in tensorboard
    # By default, it will create a new curve
    # If you specify different tb_log_name in subsequent runs, you will have split graphs
    # See: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", callback=callback) # TODO: tb_log_name with timestamp, i.e. PPO-{int(time.time())}
    model.save(os.path.join(models_dir, str(TIMESTEPS * i)))
    print(f"Last time step: {env.last_time_step}")
    fig = env.render()

env.close()

#fig.show()
#results_plotter.plot_results([log_dir], None, results_plotter.X_EPISODES, "JOB-SHOP")
#plt.show()
plot_reward_per_episode(log_dir, title='PPO')

