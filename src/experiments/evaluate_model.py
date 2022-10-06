from sb3_contrib.ppo_mask import MaskablePPO
import numpy as np
import gym
import src.envs.JobShopEnv.envs.JssEnv
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib.common.wrappers import ActionMasker
from src.wrappers import JobShopMonitor
#from sb3_contrib.common.maskable.evaluation import evaluate_policy
from src.utils import evaluate_policy_with_makespan

# See: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/docs/modules/ppo_mask.rst
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_legal_actions()

def make_jobshop_env(rank=0, seed=0, instance_name="taillard/ta41.txt", monitor_log_path=None):
    """
    Utility function for multiprocessed env.

    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    env = gym.make('jss-v1', env_config={'instance_path': f"./data/instances/{instance_name}", 'shuffle_instance': False})
    # Important: use a different seed for each environment
    env.seed(seed + rank)
    set_random_seed(seed)

    env = ActionMasker(env, mask_fn)
    # Info on monitor.csv
    # ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
    env = JobShopMonitor(env=env, filename=monitor_log_path)
    return env

def create_model(model_name="MaskablePPO", policy="MlpPolicy", env=None, n_env=1, n_steps=20, n_episodes=100, log_dir=None, verbose=1):
        model = MaskablePPO('MultiInputPolicy', env, verbose=verbose, tensorboard_log=log_dir)
        #model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
        return model



model = MaskablePPO.load("models/jss/PPO/best_model_not_tuned_25k.zip", print_system_info=True)

'''
for i in range(41, 50):
    env = make_jobshop_env(rank=0, seed=1, instance_name=f"taillard/ta{i}.txt", monitor_log_path=None)
    # required before you can step through the environment
    env.reset()

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=False, return_episode_rewards=True)
    print(f"ta{i}.txt: {mean_reward} +/- {std_reward}")

    env.close()
'''

env = make_jobshop_env(instance_name="taillard/ta41.txt")

mean_reward, std_reward, mean_makespan, std_makespan = evaluate_policy_with_makespan(model, env, n_eval_episodes=1, deterministic=True)

print(f"Mean reward: {mean_reward}\nStd reward: {std_reward}\nMean makespan: {mean_makespan}\nStd makespan: {std_makespan}")