import gym
from stable_baselines3 import PPO
import os
import time
from snakeenv import SnakeEnv



models_dir = f"models/PPO-{int(time.time())}"
log_dir = f"logs/PPO-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create the environment
env = SnakeEnv()

# required before you can step through the environment
env.reset()


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
for i in range(1, 30):
    # Pass reset_num_timesteps=False to continue the training curve in tensorboard
    # By default, it will create a new curve
    # If you specify different tb_log_name in subsequent runs, you will have split graphs
    # See: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO") # TODO: tb_log_name with timestamp, i.e. PPO-{int(time.time())}
    model.save(os.path.join(models_dir, str(TIMESTEPS * i)))


env.close()