from stable_baselines3.common.callbacks import BaseCallback
import gym
import numpy as np
from src.utils import evaluate_policy_with_makespan

class SaveBestModelCallback(BaseCallback):
    def __init__(
        self,
        check_freq: int,
        eval_env: gym.Env,
        best_model_save_path: str,
        use_episodes: bool = False,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_model_save_path = best_model_save_path
        self.eval_env = eval_env
        self.use_episodes = use_episodes
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.episode_count = 0

    def _on_step(self) -> bool:
        '''
        if self.use_episodes:
            if "done" in self.locals and self.locals["done"]:
                self.episode_count += 1
            if self.episode_count % self.check_freq == 0:
                self.evaluate_and_save()
        else:
            if self.n_calls % self.check_freq == 0:
                self.evaluate_and_save()
        return True
        '''
        return True

    def _on_rollout_end(self) -> None:
        self.evaluate_and_save()

    def evaluate_and_save(self):
        # evaluate_policy is using deterministic actions by default
        metric_dict = evaluate_policy_with_makespan(
            self.model, 
            self.eval_env, 
            n_eval_episodes=self.n_eval_episodes, 
            deterministic=self.deterministic
        )

        mean_reward = metric_dict["mean_reward"]
        mean_makespan = metric_dict["mean_makespan"]
        
        if mean_reward > self.best_mean_reward:
            print(f"Found new best model with mean reward {mean_reward} and mean makespan {mean_makespan}")
            self.best_mean_reward = mean_reward
            self.model.save(f"{self.best_model_save_path}/best_model")