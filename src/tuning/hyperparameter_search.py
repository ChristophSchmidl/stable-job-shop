import os
import random
import pprint

import numpy as np
import wandb
import torch
from wandb.integration.sb3 import WandbCallback

from src.utils.pytorch_utils import get_device, get_device_name, get_device_count, get_device_memory, enforce_deterministic_behavior
from src.models.environment import make_jobshop_env

# Make sure we always get the same "random" numbers
enforce_deterministic_behavior()
device = get_device()

def make_model(config):
    pass


def train():
    with wandb.init() as run:
        config = wandb.config

        # Create the environment
        env = make_jobshop_env(rank=0, seed=1, instance_name="taillard/ta41.txt", monitor_log_path=log_dir)
        # required before you can step through the environment
        env.reset()

        # Create the model with callbacks
        model, callback = create_model(
            model_name="MaskablePPO", 
            policy="MlpPolicy", 
            env=env, 
            n_env=1, 
            n_steps=20, 
            n_episodes=25000, 
            log_dir=log_dir, 
            verbose=1)

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True, tb_log_name="PPO", callback=callback, use_masking=True)

    model.learn(total_timesteps=episodes, reset_num_timesteps=True, tb_log_name="PPO", callback=WandbCallback(), use_masking=True) # TODO: tb_log_name with timestamp, i.e. PPO-{int(time.time())}
    model.save(os.path.join(wandb.run.dir, "models", str(episodes * i)))

    # Get the episode data
    episode_rewards = env.get_episode_rewards()
    episode_lengths = env.get_episode_lengths()
    episode_times = env.get_episode_times()
    episode_makespans = env.get_episode_makespans()

    # Log the episode data
    wandb.log({
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_times": episode_times,
        "episode_makespans": episode_makespans
    })

    # Log the model
    wandb.save(os.path.join(wandb.run.dir, "models", str(episodes * i)))

    # Log the environment
    wandb.save(os.path.join(wandb.run.dir, "env", str(episodes * i)))


'''
layer_size: 264
num_sgd_iter: 12 -> SB3 PPO: n_epochs?
episode_reward_mean: 179.046
'''


sweep_config = {
    "name": "jss-ppo-sweep",
    "method": "random",
    "metric": {
        "name": "episode_rewards",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {
            "values": [0.0008, 0.00009]
        },
        "clip_range": {
            "values": [0.1, 0.6]
        },
        "clip_range_vf": {
            "values": [0.1, 24]
        },
        "gamma": {
            "values": [0.99, 0.999]
        },
        "vf_coef": {
            "values": [0.5, 0.9999]
        },
        "ent_coef": {
            "values": [0.002, 0.006]
        },
        "target_kl": {
            "values": [0.088, 0.116]
        },
        "optimizer": {
            "values": ["adam", "sgd"]
        },
    }
}

wandb.init(config=sweep_config)
config = wandb.config



sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train)



env = make_jobshop_env(rank=0, seed=1, instance_name="taillard/ta41.txt", monitor_log_path=run.dir)


policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=[256, 256]
    lr_schedule=0.00025,
    optimizer_kwargs=dict(eps=1e-5),
    )



