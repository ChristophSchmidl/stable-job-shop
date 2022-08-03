import gym
from stable_baselines3 import A2C

# Create the environment
env = gym.make("LunarLander-v2") # Continuous control task

# required before you can step through the environment
env.reset()


model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        # pass observation to model to get predicted action
        action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action) # sample random action
        # obs, reward, done, info = env.step(env.action_space.sample()) # sample random action
        env.render() # shows the animation
        print(reward)

env.close()