import random
import numpy as np
import gym
import JSSEnv


def FIFO_worker(instance_name = "taillard/ta41.txt", seed=1337):
    print(f"Creating environment...\n")
    env = gym.make('jss-v1', env_config={'instance_path': f"./data/instances/{instance_name}"})
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
        remaining_time = reshaped[:, 5]
        illegal_actions = np.invert(legal_actions)
        mask = illegal_actions * -1e8 # Go to almost zero to make the action almost impossible?
        remaining_time += mask
        FIFO_action = np.argmax(remaining_time)
        assert legal_actions[FIFO_action]
        state, reward, done, _ = env.step(FIFO_action)
    env.reset()
    make_span = env.last_time_step
    print(make_span)
    #wandb.log({"nb_episodes": 1, "make_span": make_span})

if __name__ == "__main__":
    FIFO_worker()