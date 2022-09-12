import gym
import numpy as np

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.utils import set_random_seed
from src.wrappers.JobShopMonitor import JobShopMonitor


# See: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/docs/modules/ppo_mask.rst
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_legal_actions()

def make_jobshop_env(rank=0, seed=0, instance_name="taillard/ta01.txt", monitor_log_path=None):
    """
    Utility function for multiprocessed env.

    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    env = gym.make('jss-v1', env_config={'instance_path': f"./data/instances/{instance_name}"})
    # Important: use a different seed for each environment
    env.seed(seed + rank)
    set_random_seed(seed)

    env = ActionMasker(env, mask_fn)
    # Info on monitor.csv
    # ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
    env = JobShopMonitor(env=env, filename=monitor_log_path)
    return env