from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np
from typing import Dict, Any

class CollectExperiencesCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
 
    def __init__(self, _locals: Dict[str, Any], _globals: Dict[str, Any], verbose: int =0):
        super(CollectExperiencesCallback, self).__init__(verbose)
        self.locals = _locals
        self.globals = _globals
        self.episode = 0

    def _init_callback(self) -> None:
        self._init_callback()

    def _on_training_start(self):
        print("Starting to train...")
        

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        print("Starting to collect experiences...")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        print("Stepping through the environment...")

        #self.logger.record('rollout/episode_reward', self.locals["rewards"][0])
        #self.logger.record('rollout/test', 42)
        #episode_reward_dict = {"episode": self.episode_count, "reward": self.locals["rewards"][0]}
        #self.rewards_df = self.rewards_df.append(episode_reward_dict, ignore_index=True)


        #self.episode_rewards.append(self.locals["rewards"][0])
        #print(f"Episode rewards: { self.training_env.get_episode_rewards() }")

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        print("Collecting experiences is done!")

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        #rint("Training ends")
        #print(np.mean(self.episode_rewards))
        #self.logger.record('rollout/episode_reward', np.mean(self.episode_rewards))