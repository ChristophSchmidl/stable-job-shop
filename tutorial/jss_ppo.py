import gym
from stable_baselines3 import PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.logger import configure
import os
import time
import JSSEnv
import numpy as np


tmp_path = f"logs/sb3_log/"
models_dir = "models/jss/PPO"
log_dir = f"logs/jss/PPO-{int(time.time())}"

# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# See: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/docs/modules/ppo_mask.rst
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_legal_actions()

# Create the environment
env = gym.make('jss-v1', env_config={'instance_path': './tutorial/instances/taillard/ta41.txt'})
env = ActionMasker(env, mask_fn)

# required before you can step through the environment
env.reset()

model = MaskablePPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir)
#model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

# Set new logger
model.set_logger(new_logger)

TIMESTEPS = 100000
for i in range(1, 5):
    # Pass reset_num_timesteps=False to continue the training curve in tensorboard
    # By default, it will create a new curve
    # If you specify different tb_log_name in subsequent runs, you will have split graphs
    # See: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO") # TODO: tb_log_name with timestamp, i.e. PPO-{int(time.time())}
    model.save(os.path.join(models_dir, str(TIMESTEPS * i)))
    print(f"Last time step: {env.last_time_step}")
    env.render().show()

env.close()