import time
from stable_baselines3.common.callbacks import BaseCallback

class TimeLimitCallback(BaseCallback):
    def __init__(self, time_limit, verbose=0):
        super(TimeLimitCallback, self).__init__(verbose)
        self.time_limit = time_limit
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit:
            print(f"Training stopped: time limit of {self.time_limit} seconds reached")
            return False
        return True