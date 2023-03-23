import random
import numpy as np
import gym
import src.envs.JobShopEnv.envs.JssEnv


'''
#######################
a4) Total left-over time: The left-over time until total completion
    of the job, scaled by the longest job total completion time 
    to be in the tange [0,1]
#######################

- What are the optimal makespans? We can use Constraint Programming (CP) for that
- Dispatching rules that are often used in the literature (taken from .....)
    1. Shortest processing time (SPT)
    2. Longest processing time (LPT)
    3. Shortest total processing time (STPT)
    4. Longest total processing time (LTPT)
    5. Shortest tail processing time (STT)
    6. Longest tail processing time (LTT)
    7. Least operation remaining (LOR)
    8. Most operation remaining (MOR)
    
    9. Random selection

    10. NEH algorithm
'''

class GymWrapper:
    def __init__(self, instance_path="taillard/ta41.txt", seed=1337, perm_mode=None):
        self.instance_path = instance_path
        self.seed = seed
        self.permutation_mode = perm_mode

    def __call__(self, dispatching_rule_func):
        def wrapper(*args, **kwargs):
            print(f"Creating environment...\n")
            env = gym.make('jss-v1', 
            env_config={'instance_path': f"./data/instances/{self.instance_path}",
            'permutation_mode': self.permutation_mode})

            env.seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)

            # Start of environment loop
            print(f"Resetting environment...\n")
            state = env.reset()
            done = False
            rewards = np.array()

            while not done:
                action = dispatching_rule_func(state, env)
                state, reward, done, _ = env.step(action)

            env.close()
            make_span = env.last_time_step

            return make_span
        return wrapper
    
    





class Dispatcher:
    def __init__(self, instance_path="taillard/ta41.txt", seed=1337, perm_mode=None):
        self.instance_path = instance_path
        self.seed = seed
        self.permutation_mode = perm_mode
    
    @GymWrapper(instance_path, 1337, None)
    def fifo(self):
        pass

    def dispatch(self, rule=)
        



def FIFO_worker(instance_name = "taillard/ta41.txt", seed=1337):
    '''
    Excerpt from "A Reinforcement Learning Environment For Job-Shop Scheduling": 

    The First In First Out (FIFO) rule amounts to take the biggest value of the a_6 attribute, i.e.,
    the job which was idle for the most time since its last operation. 

    See: reshaped[:, 5] in the code below. # Give me the idle times of all jobs
    '''
    print(f"Creating environment...\n")
    env = gym.make('jss-v1', env_config={'instance_path': f"./data/instances/{instance_name}", 'permutation_mode': None})
    
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    done = False
    print(f"Resetting environment...\n")
    state = env.reset()

    while not done:
        real_state = np.copy(state['real_obs']) # shape (30,7)
        legal_actions = state['action_mask'][:-1]
        reshaped = np.reshape(real_state, (env.jobs, 7)) # shape (30,7), this does nothing?
        
        remaining_time = reshaped[:, 5] # shape (30,), a 1-d array indicating the remaining time of every job?
        illegal_actions = np.invert(legal_actions) # inverts the boolean legal_actions: True -> False, False -> True
        mask = illegal_actions * -1e8 # Go to almost zero to make the action almost impossible? The mask is a 1-d array of -0.0 ?
        remaining_time += mask
        FIFO_action = np.argmax(remaining_time) # returns the index of the largest element in remaining_time?
        assert legal_actions[FIFO_action]
        state, reward, done, _ = env.step(FIFO_action)
    env.reset()
    make_span = env.last_time_step
    
    return make_span
    #wandb.log({"nb_episodes": 1, "make_span": make_span})

def MWKR_worker(instance_name = "taillard/ta41.txt", seed=1337):
    '''
    Excerpt from "A Reinforcement Learning Environment For Job-Shop Scheduling": 

    The Most Work Remaining (MWKR) is equivalent to take the job with the largest value of the a4 attribute, 
    i.e., the job that has the most left-over time until completion.

    See: reshaped[:, 3] in the code below. # Give me the left-over time of all jobs
    '''
    print(f"Creating environment...\n")
    env = gym.make('jss-v1', env_config={'instance_path': f"./data/instances/{instance_name}", 'permutation_mode': None})
    
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    done = False
    print(f"Resetting environment...\n")
    state = env.reset()

    while not done:
        real_state = np.copy(state['real_obs'])
        legal_actions = state['action_mask'][:-1]
        reshaped = np.reshape(real_state, (env.jobs, 7))

        remaining_time = (reshaped[:, 3] * env.max_time_jobs) / env.jobs_length # env.jobs_length is affected by the permutation
        illegal_actions = np.invert(legal_actions)

        mask = illegal_actions * 1e8
        remaining_time += mask
        MWKR_action = np.argmin(remaining_time)
        assert legal_actions[MWKR_action]
        state, reward, done, _ = env.step(MWKR_action)
    env.reset()

    make_span = env.last_time_step
    
    #wandb.log({"nb_episodes": 1, "make_span": make_span})
    return make_span

def LWKR_worker(instance_name = "taillard/ta41.txt", seed=1337):
    '''
    Excerpt from "A Reinforcement Learning Environment For Job-Shop Scheduling": 

    The Least Work Remaining (MWKR) is equivalent to take the job with the lowest value of the a4 attribute, 
    i.e., the job that has the least left-over time until completion.

    See: reshaped[:, 3] in the code below. # Give me the left-over time of all jobs
    '''
    print(f"Creating environment...\n")
    env = gym.make('jss-v1', env_config={'instance_path': f"./data/instances/{instance_name}", 'permutation_mode': None})
    
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    done = False
    print(f"Resetting environment...\n")
    state = env.reset()

    while not done:
        real_state = np.copy(state['real_obs'])
        legal_actions = state['action_mask'][:-1]
        reshaped = np.reshape(real_state, (env.jobs, 7))

        remaining_time = (reshaped[:, 3] * env.max_time_jobs) / env.jobs_length # env.jobs_length is affected by the permutation
        illegal_actions = np.ma.masked_array(np.invert(legal_actions))
        # True means invaid, False means valid
        # This is following the same convention as numpy.ma
        masked_remaining_time = np.ma.masked_array(remaining_time, mask=illegal_actions)

        # mask = illegal_actions * 1e8
        #remaining_time += mask
        MWKR_action = np.ma.argmax(masked_remaining_time)
        
        assert legal_actions[MWKR_action]
        state, reward, done, _ = env.step(MWKR_action)
    env.reset()

    make_span = env.last_time_step
    #print(make_span)
    #wandb.log({"nb_episodes": 1, "make_span": make_span})
    return make_span

def RANDOM_worker(instance_name = "taillard/ta41.txt", seed=1337):
    '''
    Excerpt from "A Reinforcement Learning Environment For Job-Shop Scheduling": 

    See: reshaped[:, 3] in the code below. # Give me the left-over time of all jobs
    '''
    print(f"Creating environment...\n")
    env = gym.make('jss-v1', env_config={'instance_path': f"./data/instances/{instance_name}", 'permutation_mode': None})
    
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    done = False
    print(f"Resetting environment...\n")
    state = env.reset()

    while not done:
        real_state = np.copy(state['real_obs'])
        legal_actions = state['action_mask'][:-1]
        reshaped = np.reshape(real_state, (env.jobs, 7))

        remaining_time = (reshaped[:, 3] * env.max_time_jobs) / env.jobs_length # env.jobs_length is affected by the permutation
        illegal_actions = np.ma.masked_array(np.invert(legal_actions))
        # True means invaid, False means valid
        # This is following the same convention as numpy.ma
        masked_remaining_time = np.ma.masked_array(remaining_time, mask=illegal_actions)

        n_actions = len(illegal_actions)
        valid_actions = np.arange(n_actions)[legal_actions]
        random_action = np.random.choice(valid_actions)
        
        assert legal_actions[random_action]
        state, reward, done, _ = env.step(random_action)
    env.reset()

    make_span = env.last_time_step
    #print(make_span)
    #wandb.log({"nb_episodes": 1, "make_span": make_span})
    return make_span


if __name__ == "__main__":

    instance = "taillard/ta41.txt"

    lwkr_makespan = LWKR_worker(instance_name=instance)
    mwkr_makespan = MWKR_worker(instance_name=instance)
    fifo_makespan = FIFO_worker(instance_name=instance)
    random_makespan = RANDOM_worker(instance_name=instance)

    print(f"Makespan for instance {instance} using LWKR: {lwkr_makespan}")
    print(f"Makespan for instance {instance} using MWKR: {mwkr_makespan}")
    print(f"Makespan for instance {instance} using FIFO: {fifo_makespan}")
    print(f"Makespan for instance {instance} using RANDOM: {random_makespan}")


