import gym
import JSSEnv
import numpy as np

env = gym.make('jss-v1', env_config={'instance_path': './tutorial/instances/taillard/ta41.txt'})

average = 0

for _ in range(5):
    print()
    state = env.reset()
    legal_actions = env.get_legal_actions()
    done = False
    total_reward = 0

    machines_available = set()
    for job in range(len(env.legal_actions[:-1])):
        if env.legal_actions[job]:
            machine_needed = env.needed_machine_jobs[job]
            machines_available.add(machine_needed)

    while not done:
        actions = np.random.choice(len(legal_actions), 1, p=(legal_actions / legal_actions.sum()))[0]
        state, rewards, done, _ = env.step(actions)
        legal_actions = env.get_legal_actions()
        total_reward += rewards

        machines_available = set()
        for job in range(len(env.legal_actions[:-1])):
            if env.legal_actions[job]:
                machine_needed = env.needed_machine_jobs[job]
                machines_available.add(machine_needed)

    #env.render().show()
    print(f"Make-span: {env.last_time_step}")


    average += env.last_time_step
print(f"Total reward: {total_reward}")
print(f"Average: {average / 100}")
