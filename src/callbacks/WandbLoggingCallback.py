import wandb
from stable_baselines3.common.callbacks import BaseCallback
from src.utils import evaluate_policy_with_makespan

class WandbLoggingCallback(BaseCallback):
    def __init__(self, eval_env, n_eval_episodes, deterministic=True, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        mean_reward, std_reward, mean_makespan, std_makespan = evaluate_policy_with_makespan(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=self.deterministic)
        wandb.log({
            "mean_reward": mean_reward,
            "mean_makespan": mean_makespan,
            "total_timesteps": self.num_timesteps
        })