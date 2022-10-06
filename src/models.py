import os
from sb3_contrib.ppo_mask import MaskablePPO
from typing import Any, Dict, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import obs_as_tensor
import numpy as np
import pprint
from src.utils import permute_instance, reverse_permuted_instance

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

    def reverse_permuted_boolean_action_masks(self, action_masks, perm_matrix):
        original_action_masks = np.copy(action_masks)

        #pprint.pprint(action_masks)
        #print(f"action_masks shape: {action_masks.shape}")

        # original_action_mask[0] = the raw action mask
        # original_action_mask[0][:-1] = the raw action mask without the no_op
        permutation = reverse_permuted_instance(action_masks[0][:-1], perm_matrix)
        # Why is permutation of type tuple?
        permutation = list(permutation)
        # Append the no-op operation
        permutation.append(action_masks[0][-1])
        permutation = np.asarray(permutation, dtype=action_masks[0].dtype)
        #print(f"Type of permutation {type(permutation)}")

        original_action_masks[0] = permutation
        #pprint.pprint(original_action_masks)
        return original_action_masks

    def _reverse_permuted_real_obs(self, real_obs, perm_matrix):
        original_real_obs = np.copy(real_obs)

        # Reverse the permutation the real_obs
        permutation = reverse_permuted_instance(original_real_obs[0], perm_matrix)
        permutation = list(permutation)[0]
        permutation = np.asarray(permutation, dtype=real_obs[0].dtype)

        original_real_obs[0] = permutation
        return original_real_obs

    def _reverse_permuted_action_mask(self, action_mask, perm_matrix):
        '''
        The action_mask contains number_of_jobs + 1 entry (no_op).
        So, the permutation only has to be performed for the jobs and not the no_op.
        '''
        original_action_mask = np.copy(action_mask)

        # Reverse the permutation of actions
        permutation = reverse_permuted_instance(original_action_mask[0][:-1], perm_matrix)
        permutation = list(permutation)
        permutation.append(original_action_mask[0][-1]) # Add the no-op
        permutation = np.asarray(permutation, dtype=original_action_mask[0].dtype)
        #print(permutation)

        original_action_mask[0] = permutation
        return original_action_mask

    def reverse_permuted_action_probas(self, actions, perm_matrix):
        original_actions = np.copy(actions)

        # Reverse the permutation of actions
        permutation = reverse_permuted_instance(original_actions[0][:-1], perm_matrix)
        permutation = list(permutation)
        permutation.append(original_actions[0][-1])
        permutation = np.asarray(permutation, dtype=actions[0].dtype)
        #print(permutation)

        original_actions[0] = permutation
        return original_actions



    def reverse_permuted_observation(self, observation, perm_matrix):
        '''
        Observation is an OrderedDict with "action_mask" and "real_obs".
        action_mask: <class 'numpy.ndarray'>, shape -> (1, 31), dtype=float32 31 because number_of_jobs + 1
        real_obs: <class 'numpy.ndarray'>, shape -> (1, 30, 7), dtype=float32 30 because number_of_jobs, 7 features
        We probably have to permute both of them and update the dictionary.
        '''

        ####################################################
        #   Reversing the permutation of the action_mask
        ####################################################

        restored_action_mask = self._reverse_permuted_action_mask(observation["action_mask"], perm_matrix)
        observation["action_mask"] = restored_action_mask
        pprint.pprint(observation)


        ####################################################
        #   Handling the permutation of the real_obs
        ####################################################

        restored_real_obs = self._reverse_permuted_real_obs(observation["real_obs"], perm_matrix)

        # Update the observation
        observation["action_mask"] = restored_action_mask
        observation["real_obs"] = restored_real_obs

        return observation

      
    def _load_model(self, model_path):
        #"models/jss/PPO/best_model_not_tuned_25k.zip"
        # CHeck if file exists
        if os.path.exists(model_path):
            print("Loading model from: {}".format(model_path))
            return MaskablePPO.load(model_path, print_system_info=self.print_system_info)
        else:
            print("Model file does not exist: {}".format(model_path))

    def _print_permutation_mode(self):
        permutation_mode = self.env.get_attr('permutation_mode')[0]
        
        if permutation_mode:
            print("\n#############################################")
            print("#       Model is in permutation mode!       #")
            print("#############################################\n")
        else:
            print("\n#############################################")
            print("#       Model is NOT in permutation mode!   #")
            print("#############################################\n")

    def _predict_probas(self, model, state):
        obs = model.policy.obs_to_tensor(state)[0]
        #obs = obs_as_tensor(state, model.policy.device)
        dis = model.policy.get_distribution(obs)
        probs = dis.distribution.probs
        probs_np = probs.detach().numpy()
        return probs_np

    def _mask_probas(self, probas, mask):
        probas[~mask] = 0
        return probas


    def permute_probas(self, probas, perm_indices):
        '''
        Permute the probabilities of the actions according to the permutation indices.
        '''
        original_action_probas = np.copy(probas)

        print(f"Size of original_action_probas: {len(original_action_probas[0])}")
        pprint.pprint(original_action_probas[0])


        # Reverse the permutation of actions
        permutation, _, _ = permute_instance(original_action_probas[0][:-1], perm_indices)
        permutation = list(permutation)
        print(f"Size of permutation: {len(permutation)}")
        pprint.pprint(permutation)
        permutation.append(original_action_probas[0][-1])
        print(f"Size of permutation: {len(permutation)}")
        pprint.pprint(permutation)
        
        

        permutation = np.asarray(permutation, dtype=original_action_probas.dtype)
        #print(permutation)

        pprint.pprint(original_action_probas[0])
        pprint.pprint(permutation)

        pprint.pprint(len(original_action_probas[0]))
        pprint.pprint(len(permutation))

        original_action_probas[0] = permutation
        return original_action_probas


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

        permutation_mode = self.env.get_attr('permutation_mode')[0]
        
        # Handling inputs: observation and actions_masks
        # Input has to be reverted to the original
        if permutation_mode:

            # Handling inputs
            perm_matrix = self.env.get_attr('perm_matrix')[0]
            observation = self.reverse_permuted_observation(observation, perm_matrix)
            action_masks = self.reverse_permuted_boolean_action_masks(action_masks, perm_matrix)

            # Handling outputs
            # Output has to be permuted the same way as the Env did
            perm_indices = self.env.get_attr('perm_indices')[0]
            # 1. Get all probabilities of the original actions
            # 2. Mask the invalid actions with the reversted action_masks
            # 3. Permute the probabilities as the env did
            # 4. Get the action with the highest probability and return it

            action_distribution = self.model.policy.get_distribution(observation, action_masks)
            print(action_distribution)

            

            #action_probas = self._predict_probas(self.model, observation)
            #action_probas = self._mask_probas(action_probas, action_masks)
            #permuted_action_probas = self.permute_probas(action_probas, perm_indices)
            #_actions = [np.argmax(permuted_action_probas)]
            _states = None

        else:
            #_actions, _states = self.model.predict(observation, state, episode_start, deterministic, action_masks)
            action_distribution = self.model.policy.get_distribution(observation, action_masks)
            pprint.pprint(action_distribution)


        

        #print(permuted_action_probas)
        #index_with_hight_prob = np.argmax(permuted_action_probas)

        # 1. Apply the permuted action_mask to the action_probs
        # 1a. if index is 0, just put the value to -infty?
        # 2. Get the index of the highest value
        # https://stackoverflow.com/questions/66836922/python-filter-numpy-array-based-on-mask-array
        # 

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

