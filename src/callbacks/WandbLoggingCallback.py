import wandb
import time
from stable_baselines3.common.callbacks import BaseCallback
from src.utils import evaluate_policy_with_makespan
import numpy as np

class WandbLoggingCallback(BaseCallback):
    def __init__(self, check_freq=10, use_episodes=True, eval_env=None, n_eval_episodes=1, deterministic=True, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.use_episodes = use_episodes
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.start_time = time.time()
        self.episode_count = 0

    def _on_step(self) -> bool:
        '''
        if self.use_episodes:
            done_array = self.locals.get("dones", None)
            rewards_array = self.locals.get("rewards", None)
            if done_array is not None:
                for idx, done in enumerate(done_array):
                    if done:
                        # Log the episode reward to wandb
                        wandb.log({"reward_env_{}".format(idx): rewards_array[idx]})
                self.episode_count += 1
                
                current_reward = np.mean(self.locals['rewards'])
                wandb.log({
                    "train/episode_mean_rewards": current_reward,
                })
                
                if self.verbose > 0:
                    pass
                    #print(f"Current reward: {current_reward}")
            if self.episode_count % self.check_freq == 0:
                self.evaluate_and_log()
        else:
            if self.n_calls % self.check_freq == 0:
                self.evaluate_and_log()
        return True
        '''
        return True

    def evaluate_and_log(self) -> None:
        metric_dict = evaluate_policy_with_makespan(
            self.model, 
            self.eval_env, 
            n_eval_episodes=self.n_eval_episodes, 
            deterministic=self.deterministic
        )
        elapsed_time = time.time() - self.start_time

        wandb.log({
            "eval/min_reward": metric_dict["min_reward"],
            "eval/max_reward": metric_dict["max_reward"],
            "eval/mean_reward": metric_dict["mean_reward"],
            "eval/min_makespan": metric_dict["min_makespan"],
            "eval/max_makespan": metric_dict["max_makespan"],
            "eval/mean_makespan": metric_dict["mean_makespan"],
            "eval/elapsed_time": elapsed_time,
            "total_timesteps": self.num_timesteps
        })


    def _on_rollout_end(self) -> None:
        self.evaluate_and_log()
