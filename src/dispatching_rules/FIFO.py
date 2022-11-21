import random
import numpy as np
import gym
import src.envs.JobShopEnv.envs.JssEnv


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
    print(make_span)
    return make_span
    #wandb.log({"nb_episodes": 1, "make_span": make_span})

if __name__ == "__main__":
    FIFO_worker()