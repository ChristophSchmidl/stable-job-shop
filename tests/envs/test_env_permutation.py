from typing import Dict, Any
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv
from src.envs.JobShopEnv.envs.JssEnv import JssEnv
from src.utils.permutation_handler import PermutationHandler
from src.old_utils import make_env, evaluate_policy_with_makespan
from src.models import MaskablePPOPermutationHandler


INSTANCE_NAME="taillard/ta41.txt"
MODEL_PATH="models/jss/PPO/best_model_not_tuned_25k.zip"

def create_env( instance_name, permutation_mode):
    #return DummyVecEnv([make_env('jss-v1', 2, 456, instance_name=instance_name, permutation_mode=permutation_mode)])
    return make_env('jss-v1', None, None, instance_name=instance_name, permutation_mode=permutation_mode)()
         
def create_model(model_path, env):
    return MaskablePPOPermutationHandler(model_path=model_path, env=env, print_system_info=None)

def get_random_legal_action(legal_actions):
    return np.random.choice(np.where(legal_actions == True)[0])

def get_affected_observation_indices(data) -> np.ndarray:
    condition = np.logical_and(data >= 0.0, data != 1.0)
    #return np.where(condition)
    indices_array = np.where(np.all(condition, axis=1))
    #indices = np.argwhere(np.any(condition, axis=1))
    return indices_array

def _real_obs_equal(data1, data2):
    return np.array_equal(data1["real_obs"], data2["real_obs"])

def _action_masks_equal(data1, data2):
    return np.array_equal(data1["action_mask"], data2["action_mask"])

def test_env_reset_observation_permutation():
    '''
    1. Test if observation['real_obs'] is equal to randomlny_permuted_observation['real_obs']
    2. Test if observation['action_mask'] is equal to randomlny_permuted_observation['action_mask']

    Note: Even though there are random permutations in one env, in the beginning they are the same.

    ###########################################
    # Note: observation is a dict with keys:
    # "real_obs" and value: np.array
    # "action_mask" and value: np.array
    ###########################################
    '''
    normal_env = create_env(instance_name=INSTANCE_NAME, permutation_mode=None)
    random_env = create_env(instance_name=INSTANCE_NAME, permutation_mode="random")
 
    normal_observation = normal_env.reset() 
    randomly_permuted_observation = random_env.reset()
    
    assert _real_obs_equal(normal_observation, randomly_permuted_observation), "The real_obs are the not same"
    assert _action_masks_equal(normal_observation, randomly_permuted_observation), "The action_masks are the not same"

    perm_indices = random_env.perm_indices
    assert normal_env.perm_indices is None
    assert type(perm_indices) is np.ndarray
    
    normal_legal_actions = normal_env.get_legal_actions() # Boolean vector
    permuted_legal_actions = random_env.get_legal_actions() # Boolean vector

    # At the beginning they are the same because every job is available
    assert np.array_equal(normal_legal_actions, permuted_legal_actions), "The boolean arrays of legal actions are not equal"

def test_action_index_permutation():
    normal_env = create_env(instance_name=INSTANCE_NAME, permutation_mode=None)
    random_env = create_env(instance_name=INSTANCE_NAME, permutation_mode="random")

    normal_observations = normal_env.reset() 
    randomly_permuted_observations = random_env.reset()

    perm_indices = random_env.perm_indices

    normal_legal_actions = normal_env.get_legal_actions() # Boolean vector
    permuted_legal_actions = random_env.get_legal_actions() # Boolean vector

    #########################
    #   Permute/Inverse permute (normal action index)
    #########################
    normal_legal_action_index = get_random_legal_action(normal_legal_actions)
    assert normal_legal_actions[normal_legal_action_index] == True

    permuted_legal_action_index = PermutationHandler.get_permuted_action_index(normal_legal_action_index, perm_indices)
    assert normal_legal_action_index == PermutationHandler.get_inverse_permuted_action_index(permuted_legal_action_index, perm_indices)

    #########################
    #  Test if the action is still valid after permutation for the randomly permuted env
    #########################
    assert permuted_legal_actions[permuted_legal_action_index] == True

def test_env_step_permutation():
    normal_env = create_env(instance_name=INSTANCE_NAME, permutation_mode=None)
    random_env = create_env(instance_name=INSTANCE_NAME, permutation_mode="random")

    normal_observations = normal_env.reset() 
    randomly_permuted_observations = random_env.reset()

    perm_indices = random_env.perm_indices

    normal_legal_actions = normal_env.get_legal_actions() # Boolean vector
    permuted_legal_actions = random_env.get_legal_actions() # Boolean vector

    #########################
    #   Permute action index and revert it back
    #########################
    normal_legal_action_index = get_random_legal_action(normal_legal_actions)
    # Create action index for random_env
    permuted_legal_action_index = PermutationHandler.get_permuted_action_index(normal_legal_action_index, perm_indices)

    #########################
    #   Apply actions
    #########################

    normal_env_next_observation, normal_rewards, normal_dones, normal_infos = normal_env.step(normal_legal_action_index)
    random_env_next_observation, permuted_rewards, permuted_dones, permuted_infos = random_env.step(permuted_legal_action_index)

    # 1. Check if perm(normal_env_next_observation) == random_env_next_observation
    #    1a) Check if the real_obs are equal
    #    1b) Check if the action_mask are equal
    permuted_normal_env_next_observation = {}
    permuted_normal_env_next_observation["real_obs"], _ = PermutationHandler.permute(normal_env_next_observation["real_obs"], perm_indices)
    permuted_normal_env_next_observation["action_mask"] = PermutationHandler.permute_action_mask(normal_env_next_observation["action_mask"], perm_indices)

    print(f"Normal env next observation: {normal_env_next_observation['real_obs']}")
    print(f"Permuted normal env next observation: {permuted_normal_env_next_observation['real_obs']}")

    assert _real_obs_equal(permuted_normal_env_next_observation, random_env_next_observation), "The real_obs are the not same"
    assert _action_masks_equal(permuted_normal_env_next_observation, random_env_next_observation), "The action_masks are the not same"

    # 2. Check if inverse_perm(random_env_next_observation, perm_indices) == normal_env_next_observation
    #    2a) Check if the real_obs are equal
    #    2b) Check if the action_mask are equal
    inverse_permuted_random_env_next_observation = {}
    inverse_permuted_random_env_next_observation["real_obs"] = PermutationHandler.inverse_permute(random_env_next_observation["real_obs"], perm_indices)
    inverse_permuted_random_env_next_observation["action_mask"] = PermutationHandler.inverse_action_mask(random_env_next_observation["action_mask"], perm_indices)

    assert _real_obs_equal(inverse_permuted_random_env_next_observation, normal_env_next_observation), "The real_obs are the not same"
    assert _action_masks_equal(inverse_permuted_random_env_next_observation, normal_env_next_observation), "The action_masks are the not same"

    #print(f"Normal next observations: {normal_next_real_obs}")
    #print(f"Permuted next observations: {permuted_next_real_obs}")
    #print(f"Inverted permuted next observations: {inverted_permuted_next_real_obs}")

    # Normal next observations: the masking is correct
    # Permuted next observations: the masking is wrong
    # Inverted permuted next observations: the masking is wrong

if __name__ == "__main__":
    args = ["tests/envs", "-v", "-s", "-W", "ignore::DeprecationWarning"]
    pytest.main(args)