import gym
import JSSEnv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import pandas as pd


env = gym.make('jss-v1', env_config={'instance_path': './tutorial/instances/taillard/ta01.txt'})

EPISODES = 200



episode_rewards = []
average_episode_rewards = []
episode_makespans = []

for episode in range(EPISODES):
    print()
    state = env.reset()
    legal_actions = env.get_legal_actions()
    done = False
    cum_episode_reward = 0

    #print(env.legal_actions[:-1])

    # The whole purpose here is to get all machine indices into machines_available???
    machines_available = set() # unordered collection of unique elements
    for job in range(len(env.legal_actions[:-1])): # -1 to exclude the last action: no-op?
        if env.legal_actions[job]:
            machine_needed = env.needed_machine_jobs[job] # needed_machine_jobs = np.zeros(self.jobs, dtype=np.int_)
            '''
            env.needed_machine_jobs contains an array of the machines needed for each job.
            This probably changes every time the step() function is called.
            '''
            machines_available.add(machine_needed)

    #print(env.needed_machine_jobs)

    # Step through the environment until done
    steps = 0
    while not done:
        steps += 1
        actions = np.random.choice(len(legal_actions), 1, p=(legal_actions / legal_actions.sum()))[0]
        state, rewards, done, _ = env.step(actions)
        legal_actions = env.get_legal_actions()
        cum_episode_reward += rewards

        machines_available = set()
        for job in range(len(env.legal_actions[:-1])):
            if env.legal_actions[job]:
                machine_needed = env.needed_machine_jobs[job]
                machines_available.add(machine_needed)

        #print(env.needed_machine_jobs)
        '''
        env.needed_machine_jobs contains the indices of the machines needed for each job.
        If the index contains -1 then the job is not assigned to a machine or the job is assigned to a machine that is not available.
        [-1 12 -1 -1  1 -1  9 -1 13 -1 -1 11 14 -1  5]

        The last time step looks like this:
        [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
        '''

    #env.render().show()
    #print(f"Make-span: {env.last_time_step}")
    episode_rewards.append(cum_episode_reward)
    episode_makespans.append(env.last_time_step)
    #average_episode_rewards


print(f"Episode rewards: {episode_rewards}")
#print(f"Episode average rewards: {average_episode_rewards}")
print(f"Episode make-spans: {episode_makespans}")

data_plot = pd.DataFrame({"Episode": np.arange(1, len(episode_rewards)+1), "Episode reward":episode_rewards})
sns.lineplot(x = "Episode", y = "Episode reward", data = data_plot)
plt.show()