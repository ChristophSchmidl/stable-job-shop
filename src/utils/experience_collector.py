from typing import Dict, Any
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from src.envs.JobShopEnv.envs.JssEnv import JssEnv
from src.old_utils import make_env, evaluate_policy_with_makespan
from src.models import MaskablePPOPermutationHandler


class ExperienceCollector:
    def __init__(self, instance_name="taillard/ta41.txt", model_path="models/jss/PPO/best_model_not_tuned_25k.zip", n_episodes=10, save_every_n_episodes=50, permutation_mode="random", data_dir="data/experiences"):
        self.env = self._create_env(instance_name, permutation_mode)
        self.model = self._create_model(model_path, self.env)
        self.n_episodes = n_episodes
        self.save_every_n_episodes = save_every_n_episodes
        self.permutation_mode = permutation_mode # "random" or "transpose=1,2,3,..."
        self.data_dir = data_dir
        
        self.states = []
        self.action_masks = []
        self.actions = []
        self.dones = []
        self.episodes = []

    def _create_env(self, instance_name, permutation_mode):
        return DummyVecEnv([make_env('jss-v1', 2, 456, instance_name=instance_name, permutation_mode=permutation_mode)])
         
    def _create_model(self, model_path, env):
        return MaskablePPOPermutationHandler(model_path=model_path, env=env, print_system_info=None)

    def _get_file_name(self, suffix=None):
        perm_mode = None

        if self.permutation_mode is None:
            perm_mode = "no-permutation"
        elif self.permutation_mode == "random":
            perm_mode = "random"
        else:
            perm_mode = "transpose-" + self.permutation_mode.split("=")[1]

        if suffix is None:
            return f"experiences_{perm_mode}_{self.n_episodes}-episodes.npz"
        else:
            return f"experiences_{perm_mode}_{self.n_episodes}-episodes_{suffix}.npz"

    def _collect_experiences_callback(self, _locals: Dict[str, Any], _globals: Dict[str, Any]):
        #pprint.pprint(f"Printing _locals: {_locals}")
        #pprint.pprint(f"Printing _locals: {_globals}")
        # What we want to collect: states, actions, rewards, dones, infos
        current_episode = _locals["episode_counts"][0]
        current_step = _locals["current_lengths"][0]
        state = _locals["observations"]["real_obs"] 
        boolean_action_mask = _locals["action_masks"] # This is a boolean mask
        action = _locals["actions"] # This is the action that was actually taken. Unwrapping?
        reward = _locals["reward"]
        done = _locals["done"]
        makespan = _locals["info"]["makespan"]

        self.states.append(state[0])
        self.action_masks.append(boolean_action_mask[0])
        self.actions.append(action[0])
        self.dones.append(done)
        self.episodes.append(current_episode)

        # Save experiences to file every n episodess
        if (current_episode + 1) % self.save_every_n_episodes == 0 and done:
            self._save_experiences(current_episode, current_step, self.states, self.actions, self.action_masks, self.dones, self.episodes)
            
        # Save experiences to file when last episode is done
        if (current_episode + 1) == self.n_episodes and done:
            self._save_experiences(current_episode, current_step, self.states, self.actions, self.action_masks, self.dones, self.episodes)
            
    def _save_experiences(self, current_episode, current_step, states, actions, action_masks, dones, episodes, verbose=True):
        if verbose:
            print(f"Episode {current_episode} finished with {current_step} experiences. Saving to file...")
        np.savez(f"{self.data_dir}/{self._get_file_name()}", states=states, actions=actions, action_masks=action_masks, dones=dones, episodes=episodes)

    def _reset_experiences(self):
        self.states = []
        self.action_masks = []
        self.actions = []
        self.dones = []
        self.episodes = []

    def start(self, verbose=True):
        if verbose:
            print("\n###############################################")
            print("#       Starting to collect experiences!      #")
            print("###############################################\n")

        mean_reward, std_reward, mean_makespan, std_makespan = evaluate_policy_with_makespan(
        model=self.model, 
        env=self.env, 
        n_eval_episodes=self.n_episodes,
        deterministic=True,
        callback=self._collect_experiences_callback,
        use_masking=True
        )

        if verbose:
            print("\n###############################################")
            print("#       Finished to collect experiences!      #")
            print("###############################################\n")

        print(f"Mean reward: {mean_reward}\nStd reward: {std_reward}\nMean makespan: {mean_makespan}\nStd makespan: {std_makespan}")