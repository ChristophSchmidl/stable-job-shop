import random
import gym
import numpy as np
import src.envs.JobShopEnv.envs.JssEnv


def MWKR_worker(instance_name = "taillard/ta41.txt", seed=1337):
    '''
    Most Work Remaining (MWKR)

    The Most Work Remaining (MWKR) is equivalent to take the job with the largest value of the a4 attribute, 
    i.e., the job that has the most left-over time until completion.
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
    print(make_span)
    #wandb.log({"nb_episodes": 1, "make_span": make_span})
    return make_span


if __name__ == "__main__":
    MWKR_worker()