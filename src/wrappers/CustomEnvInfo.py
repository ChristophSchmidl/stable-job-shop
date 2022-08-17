import gym

class CustomEnvInfo(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = super().step(action)
        info.update({"makespan": self.current_time_step})
        return obs, rew, done, info