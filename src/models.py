import os
from sb3_contrib.ppo_mask import MaskablePPO
from typing import Any, Dict, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
import numpy as np
import pprint
from src.utils.permutation_handler import PermutationHandler

###############################################################
#                       Operation class
###############################################################

'''
This class is not really used atm but it is here for future reference.
'''
class Operation:
  def __init__(self, job_id, machine_id, duration):
    self.job_id = job_id # Integer
    self.machine_id = machine_id # Integer
    self.duration = duration # Integer

  def __repr__(self):
    return "Job ID: {},\nMachine ID: {},\nDuration: {}".format(
      str(self.job_id), 
      str(self.machine_id), 
      str(self.duration))


###############################################################
#                       Job class
###############################################################

class Job:
    """
    A class that creates jobs.
    Parameters
    ----------
    r: list - A list with the task sequence
    p: list - Processing times for every task
    """

    def __init__(self, id, r = None, p = None):
        self.id = id

        if r is None:
            self.r = [] # machine_routes
        else:
            self.r = r
        
        if p is None:
            self.p = [] # processing times
        else:
            self.p = p

        # TODO
        self.d = []  # due dates

    def get_horizon(self):
        return sum(self.p)

    def get_max_processing_time(self):
        return max(self.p)

    def add_operation(self, r, p):
        self.r.append(r)
        self.p.append(p)

    def get_operation_count(self):
        return len(self.r)

    def __repr__(self) -> str:
        return "Id: {},\nMachine routes: {},\nProcessing times: {}\n".format(self.id, self.r, self.p)

    def get_tuple_list_representation(self):
        return [(r, p) for r, p in zip(self.r, self.p)]

    def get_operation_as_tuple(self, operation_index):
        return (self.r[operation_index], self.p[operation_index])

    def get_machine_route_list(self):
        return self.r

    def get_processing_time_list(self):
        return self.p


###############################################################
#                    JobShopInstance class
###############################################################

class JobShopInstance:
  def __init__(self, filename, name, job_count, machine_count):
      self.filename = filename
      self.name = name
      self.job_count = job_count
      self.machine_count = machine_count
      self.jobs = []
    
  def add_job(self, job):
      self.jobs.append(job)

  def get_horizon(self):
      '''
      The horizon is the sum of all operation durations. 
      It gives you the worst makespan if every job is scheduled after each other.
      '''
      return sum(job.get_horizon() for job in self.jobs)

  def get_max_processing_time(self):
        return max(job.get_max_processing_time() for job in self.jobs)

  def get_operation_count(self):
      return sum(job.get_operation_count() for job in self.jobs)

  def get_operation(self, job_id, operation_index):
      #return self.all_operations[job_id][operation_index]
      #return self.jobs[job_id][operation_index]
      first = next(filter(lambda job: job.job_id == job_id, self.jobs), None)
      return first[operation_index]

  def get_job(self, job_id):
      return self.jobs[job_id]    

  def print_report(self):
      print("\nJob-shop problem instance in JSSP format read from file: {}".format(self.filename))
      print("Name: {}".format(self.name))
      print("Number of jobs: {}".format(self.job_count))
      print("Number of machines: {}".format(self.machine_count))
      print("Number of operations: {}".format(self.get_operation_count()))
      print("Horizon (duration sum): {}".format(self.get_horizon()))
      print("==========================================\n")

      '''
      for idx, job in enumerate(self.all_operations):
          print("Job: {}\n".format(idx))
          for operation in job:
              print((operation.machine_id, operation.duration))
          print("\n")
      '''

  def get_jobs_as_list_of_tuples_with_job_id(self):
      return [(job.Id, job.get_tuple_list_representation()) for job in self.jobs]

  def get_jobs_as_list_of_tuples(self):
      return [job.get_tuple_list_representation() for job in self.jobs]

  def print_raw_contents(self):
      pass


###############################################################
#              MaskablePPOPermutationHandler class
###############################################################

class MaskablePPOPermutationHandler:
    def __init__(self, model_path, env, print_system_info):
      self.model_path = model_path
      self.env = env
      self.print_system_info = print_system_info
      self.model = self._load_model(model_path)
      self._print_permutation_mode()




    def reverse_permuted_action_probas(self, actions, perm_matrix):
        original_actions = np.copy(actions)

        # Reverse the permutation of actions
        permutation = PermutationHandler.inverse_permute(original_actions[0][:-1], perm_matrix)
        permutation = list(permutation)
        permutation.append(original_actions[0][-1])
        permutation = np.asarray(permutation, dtype=actions[0].dtype)
        #print(permutation)

        original_actions[0] = permutation
        return original_actions



      
    def _load_model(self, model_path):
        #"models/jss/PPO/best_model_not_tuned_25k.zip"
        # CHeck if file exists
        if os.path.exists(model_path):
            print("Loading model from: {}".format(model_path))
            return MaskablePPO.load(model_path, print_system_info=self.print_system_info)
        else:
            print("Model file does not exist: {}".format(model_path))

    def _print_permutation_mode(self):
        #permutation_mode = self.env.get_attr('permutation_mode')[0]
        permutation_mode = self.env.permutation_mode
        
        if permutation_mode is not None and permutation_mode == "random":
            print("\n##################################################")
            print("#       Model is in random permutation mode!       #")
            print("####################################################\n")
        elif permutation_mode is not None and permutation_mode.find("transpose") != -1:
            n_swaps = int(permutation_mode.split("=")[1])
            print("\n############################################")
            print(f"#       Model is in {permutation_mode} mode!      #")
            print("############################################\n")
        else:
            print("\n#############################################")
            print("#       Model is NOT in permutation mode!   #")
            print("#############################################\n")

    def _predict_masked_probas(self, model, state, action_masks=None):
        #obs = model.policy.obs_to_tensor(state)[0]
        obs = obs_as_tensor(state, model.policy.device)
        dis = model.policy.get_distribution(obs, action_masks) # I can insert the boolean mask here
        probs = dis.distribution.probs
        probs_np = probs.detach().numpy()
        return probs_np


    def permute_probas(self, probas, perm_indices):
        '''
        Permute the probabilities of the actions according to the permutation indices.
        '''
        original_action_probas = np.copy(probas)

        permutation, _, _ = PermutationHandler.permute(original_action_probas[0][:-1], perm_indices)
        permutation = list(permutation)

        permutation.append(original_action_probas[0][-1])
        permutation = np.asarray(permutation, dtype=original_action_probas.dtype)

        original_action_probas[0] = permutation
        return original_action_probas

    def invert_observation(self, observation, perm_indices):
        
        #observation = np.copy(observation)
        action_mask = observation["action_mask"]
        real_obs = observation["real_obs"]

        ####################################################
        #   Reversing the permutation of the action_mask
        ####################################################
        '''
        The action_mask contains number_of_jobs + 1 entry (no_op).
        So, the permutation only has to be performed for the jobs and not the no_op.
        '''    
        inverse_action_mask = PermutationHandler.inverse_permute(action_mask[:-1], perm_indices)
        inverse_action_mask = np.append(inverse_action_mask, action_mask[-1]).astype(action_mask.dtype) # Add the no-op
        observation["action_mask"] = inverse_action_mask

        ####################################################
        #   Handling the permutation of the real_obs
        ####################################################

        inverse_real_obs = PermutationHandler.inverse_permute(observation["real_obs"], perm_indices)
        observation["real_obs"] = inverse_real_obs

        return observation


    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """

        #permutation_mode = self.env.get_attr('permutation_mode')[0]
        permutation_mode = self.env.permutation_mode
        
        if permutation_mode is not None:
            #perm_indices = self.env.get_attr('perm_indices') # To repeat the permuation of the env
            #perm_matrix = self.env.get_attr('perm_matrix')[0] # To reverse the permutation of the env
            perm_indices = self.env.perm_indices
            
            inverted_observation = self.invert_observation(observation.copy(), perm_indices)
            inverted_action_masks = PermutationHandler.inverse_action_mask(action_masks.copy(), perm_indices)

            _actions, _states = self.model.predict(inverted_observation, state, episode_start, deterministic, inverted_action_masks)
            #print(f"Actions before permutation: {_actions}")
            _actions = [PermutationHandler.get_permuted_action_index(_actions, perm_indices)]
            #print(f"Actions after permutation: {_actions}")

        
            #action_probas = self._predict_masked_probas(self.model, observation, action_masks)
            #permuted_action_probas = self.permute_probas(action_probas, perm_indices)
            #_actions = [np.argmax(permuted_action_probas)]
        else:
            _actions, _states = self.model.predict(observation, state, episode_start, deterministic, action_masks)

        return _actions, _states


'''
1. You get the observation space as input. The observation space is already permuted.
2. You reverse the permutation based on the permutation matrix.
3. You feed the reversed observation space to the model so it fits the model's input space that it was trained on.
    -> model.predict(obs)?
4. You get the action space as output (vector of probs?). This action space is not permuted yet.
    -> How to get this from MaskablePPO?
    -> https://stackoverflow.com/questions/66428307/how-to-get-action-propability-in-stable-baselines-3
    -> Or just model.predict with Deterministic=True?
5. You permute the action space based on the permutation matrix so that it fits the original observation space.
6. You collect the experiences: https://www.reddit.com/r/reinforcementlearning/comments/wb74ck/ppo_rollout_buffer_for_turnbased_twoplayer_game/
7. Transition into next state.
'''

