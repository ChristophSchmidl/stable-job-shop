import gym
import torch
from stable_baselines3 import A2C

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')



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