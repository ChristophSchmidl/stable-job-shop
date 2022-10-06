import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import gym
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
from IPython.display import display
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib.common.wrappers import ActionMasker
from src.wrappers import JobShopMonitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from sb3_contrib.ppo_mask import MaskablePPO


def enforce_deterministic_behavior():
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

def get_device():
    # Device configuration
    # See also: https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def get_device_name():
    device = get_device()
    return torch.cuda.get_device_name(device)

def get_device_count():
    return torch.cuda.device_count()

def get_device_memory():
    device = get_device()
    return torch.cuda.get_device_properties(device).total_memory

def plot_reward_log(filename='logs/sb3_log/reward_log.csv'):
    log_df = pd.read_csv(filename)

    sns.set_style('darkgrid')

    fig, ax1 = plt.subplots()
    line_labels = ["Reward", "Makespan"]

    color1 = 'blue'
    line1 = sns.lineplot(x = "episode", y = "reward", color=color1, data=log_df)
    ax1.set_ylabel('Reward', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()

    color2 = 'orange'
    line2 = sns.lineplot(x = "episode", y= "makespan", color=color2, ax=ax2, data=log_df)
    ax2.set_ylabel('Makespan', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(False) # no grid over top of plots

    # see: https://stackoverflow.com/questions/39500265/how-to-manually-create-a-legend

    reward_line = mlines.Line2D([], [], color=color1, label='Reward')
    makespan_line = mlines.Line2D([], [], color=color2, label='Makespan')

    # FIFO - for ta41
    ax2.axhline(y=2543, color='green', marker='o', linestyle='--', linewidth = 4);
    # MWKR - for ta41
    ax2.axhline(y=2632, color='red', marker='o', linestyle='--', linewidth = 4);

    fifo_line = mlines.Line2D([], [], color="green", label='FIFO (2543)')
    mwkr_line = mlines.Line2D([], [], color="red", label='MWKR (2632)')

    #reward_patch = mpatches.Patch(color=color1, label='Reward')
    #makespan_patch = mpatches.Patch(color=color2, label='Makespam')

    plt.legend(handles=[reward_line, makespan_line, fifo_line, mwkr_line], loc='lower left')

    #ax2.legend(loc="upper right");
    #plt.legend(labels=labels, loc='upper right')

    plt.show()

def display_df(filename='logs/sb3_log/monitor.csv'):
    df = pd.read_csv()
    #print(df.to_string()) 
    display(df)

def create_agent(algorithm="MaskablePPO", policy="MultiInputPolicy", policy_kwargs=None, env=None, log_dir=None, verbose=1):
    if algorithm == "MaskablePPO":
        #stopTrainingOnMaxEpisodes_callback = StopTrainingOnMaxEpisodes(max_episodes = n_episodes, verbose=verbose)
        #tensorboard_callback = TensorboardCallback()
        #saveOnBestTrainingReward_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, model_dir=models_dir, verbose=verbose)
        '''
        eval_callback = EvalCallback(env, best_model_save_path='models/jss/PPO/best_model',
                             log_path=log_dir, eval_freq=5,
                             deterministic=False, render=False)
        '''
        # Create the callback list
        #callback = CallbackList([stopTrainingOnMaxEpisodes_callback, saveOnBestTrainingReward_callback, tensorboard_callback])

        model = MaskablePPO(
            policy='MultiInputPolicy', # alias of MaskableMultiInputActorCriticPolicy
            env=env, 
            policy_kwargs=policy_kwargs,
            verbose=verbose, 
            tensorboard_log=log_dir)
        #model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
        return model

# See: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/docs/modules/ppo_mask.rst
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_legal_actions()

def make_env(env_id, rank=0, seed=0, instance_name="taillard/ta01.txt", permutation_mode=False, permutation_matrix = None, monitor_log_path=None):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, env_config={"instance_path": f"./data/instances/{instance_name}", "permutation_mode": permutation_mode, "permutation_matrix": permutation_matrix})
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        
        if env_id == "jss-v1":
            print("Connecting ActionMasker and JobShopMonitor...\n")
            env = ActionMasker(env, mask_fn)
            #env = JobShopMonitor(env=env, filename=monitor_log_path) # None means, no log file
            #env = VecMonitor(env, monitor_log_path) # None means, no log file

        return env
    set_random_seed(seed)
    return _init

def permute_instance(instance, perm_indices=None):
  '''
  Returns the permuted instance and the permutation indices (for repeating the same permutation)
  '''
  # see also: https://stackoverflow.com/questions/18272160/access-multiple-elements-of-list-knowing-their-index
  job_count = len(instance)
  perm_indices = perm_indices
  permuted_instance = None

  if perm_indices is None:
    # Do random permutation
    #print("Doing random permutation...")
    perm_indices = torch.randperm(job_count).long()
    perm_matrix = torch.eye(job_count)[perm_indices].t()
  else:
    # Do permutation based on perm_indices
    #print("Doing permutation based on indices...")
    perm_matrix = torch.eye(job_count)[perm_indices].t()

  #print(perm_indices)
  permuted_instance = [ instance[i] for i in perm_indices]
  #print(f"Perm_indices type {type(perm_indices)}")
  #print(f"Perm_indices shape {perm_indices.shape}")
  #print(f"Permuted instance type {type(permuted_instance)}")
  #print(f"Item of permuted instance type {type(permuted_instance[0])}")
#  print(f"Item content of permuted instance type {permuted_instance[0][0]}")

  # Permuted_instance is a list of torch tensors which we do not want?
  return permuted_instance, perm_matrix, perm_indices

def reverse_permuted_instance(permuted_instance, perm_matrix):
  '''
  '''
  job_count = len(permuted_instance)
  restore_indices = torch.Tensor(list(range(job_count))).view(job_count, 1)
  restore_indices = perm_matrix.mm(restore_indices).view(job_count).long()
  restored_instance = [ permuted_instance[i] for i in restore_indices]
  print(f"Restore indices type {type(restore_indices)}")
  print(f"Restore indices shape {restore_indices.shape}")
  print(f"Restored instance type {type(restored_instance)}")
  return restored_instance


'''
This one is taken from sb3-contrib with masking support and has been tweaked to work with makespan
'''
def evaluate_policy_with_makespan(  # noqa: C901
    model: MaskablePPO,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    use_masking: bool = True,
    ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:

    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episde lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :param use_masking: Whether or not to use invalid action masks during evaluation
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """

    if use_masking and not is_masking_supported(env):
        raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

    is_monitor_wrapped = False

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episodes_makespans = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")

    observations = env.reset()
    states = None

    while (episode_counts < episode_count_targets).any():
        if use_masking:
            action_masks = get_action_masks(env)
            print(f"Starting prediction in the evaluate policy with makespan function...") 
            actions, state = model.predict(
                observations,
                state=states,
                deterministic=deterministic,
                action_masks=action_masks,
            )
            print("Prediction done")
        else:
            actions, states = model.predict(observations, state=states, deterministic=deterministic)
        print(f"Starting the step function...")
        observations, rewards, dones, infos = env.step(actions)
        print("Step function done")
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            #episodes_makespans.append(info["episode"]["m"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episodes_makespans.append(info["makespan"])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    if states is not None:
                        states[i] *= 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    mean_makespan = np.mean(episodes_makespans)
    std_makespan = np.std(episodes_makespans)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward, mean_makespan, std_makespan