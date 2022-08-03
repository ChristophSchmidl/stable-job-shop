from snakeenv import SnakeEnv

env = SnakeEnv()
episodes = 50

for episode in range(episodes):
    done = False
    obs = env.reset()

    print(f"Shape of observation of reset: {obs}")
    print(f"Type of observation of reset: {obs.dtype}")

    while not done:
        random_action = env.action_space.sample()
        print("action ", random_action)
        obs, reward, done, info = env.step(random_action)
        print("reward ", reward)