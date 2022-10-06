from gym.envs.registration import register

print("Registering jss-v1")

register(
    id='jss-v1', # id: The string used to create the environment with `gym.make`
    entry_point='src.envs.JobShopEnv.envs:JssEnv', # entry_point: The location of the environment to create from
)