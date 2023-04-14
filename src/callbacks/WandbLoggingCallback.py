import wandb
import time
from stable_baselines3.common.callbacks import BaseCallback
from src.utils import evaluate_policy_with_makespan

class WandbLoggingCallback(BaseCallback):
    def __init__(self, eval_env, n_eval_episodes, deterministic=True, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.start_time = time.time()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        metric_dict = evaluate_policy_with_makespan(
            self.model, 
            self.eval_env, 
            n_eval_episodes=self.n_eval_episodes, 
            deterministic=self.deterministic
        )
        elapsed_time = time.time() - self.start_time

        wandb.log({
            "min_reward": metric_dict["min_reward"],
            "max_reward": metric_dict["max_reward"],
            "mean_reward": metric_dict["mean_reward"],
            "min_makespan": metric_dict["min_makespan"],
            "max_makespan": metric_dict["max_makespan"],
            "mean_makespan": metric_dict["mean_makespan"],
            "elapsed_time": elapsed_time,
            "total_timesteps": self.num_timesteps
        })
