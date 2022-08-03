import gym
from stable_baselines3 import PPO
import os

models_dir = "models/ppo"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create the environment
env = gym.make("LunarLander-v2") # Continuous control task

# required before you can step through the environment
env.reset()


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="ppo")
    model.save(os.path.join(models_dir, "ppo_" + str(TIMESTEPS * i)))


env.close()