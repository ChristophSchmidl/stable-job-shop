import bisect
import datetime
import random

import pandas as pd
import gym
import numpy as np
import plotly.figure_factory as ff
from pathlib import Path
import sys
import inspect

from src.io.jobshoploader import JobShopLoader
from src.permutation_handler import PermutationHandler


class JssEnv(gym.Env):

    def __init__(self, env_config=None):
        """
        This environment model the job shop scheduling problem as a single agent problem:

        -The actions correspond to a job allocation + one action for no allocation at this time step (NOPE action)

        -We keep a time with next possible time steps

        -Each time we allocate a job, the end of the job is added to the stack of time steps

        -If we don't have a legal action (i.e. we can't allocate a job),
        we automatically go to the next time step until we have a legal action

        -
        :param env_config: Ray dictionary of config parameter
        """

        ###########################
        # Possible permutation modes: None, "random", "transpose={int}"
        ###########################

        if env_config is None:
            #env_config = {'instance_path': str(Path(__file__).parent.absolute()) + '/instances/ta80'}
            env_config={'instance_path': f"./data/instances/taillard/ta41.txt"}
            env_config={'permutation_mode': None}


        instance_path = env_config['instance_path']
        self.permutation_mode = env_config['permutation_mode']
        self.perm_matrix = None
        self.perm_indices = None
        # initial values for variables used for instance
        self.jobs = 0
        self.machines = 0
        ############################################
        #   Permutation relevant properties: Begin
        ############################################
        self.instance_matrix = None # This is the instance matrix after permutation if permutation_mode is True
        self.original_instance_matrix = None    # This is the original instance matrix without permutation

        self.jobs_length = None
        self.original_jobs_length = None
        ############################################
        #   Permutation relevant properties: End
        ############################################

        self.max_time_op = 0
        self.max_time_jobs = 0
        self.nb_legal_actions = 0
        self.nb_machine_legal = 0
        # initial values for variables used for solving (to reinitialize when reset() is called)
        self.solution = None
        self.last_time_step = float('inf')
        self.current_time_step = float('inf')
        self.next_time_step = list()
        self.next_jobs = list()
        self.legal_actions = None
        self.time_until_available_machine = None
        self.time_until_finish_current_op_jobs = None
        self.todo_time_step_job = None
        self.total_perform_op_time_jobs = None
        self.needed_machine_jobs = None
        self.total_idle_time_jobs = None
        self.idle_time_jobs_last_op = None
        self.state = None
        self.illegal_actions = None
        self.action_illegal_no_op = None
        self.machine_legal = None
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()
        self.sum_op = 0 # summed time of all operations

        ###################################################
        #                   Load instance                     
        ################################################### 

        loaded_instance = JobShopLoader.load_jobshop(instance_path)

        self.jobs = loaded_instance.job_count # Total number of jobs
        self.machines = loaded_instance.machine_count # Total number of machines
        # matrix which stores tuples of (machine, length of the job)
        self.original_instance_matrix = np.zeros((self.jobs, self.machines), dtype=(np.int_, 2))
        # contains all the time to complete jobs
        self.original_jobs_length = np.zeros(self.jobs, dtype=np.int_)
        self.max_time_op = loaded_instance.get_max_processing_time() # Check if the operation time is the max operation time
        self.sum_op = loaded_instance.get_horizon() # Sum of all operations
        
        # Get the summed processing time of each job
        for job in loaded_instance.jobs:
            self.original_jobs_length[job.id] = job.get_horizon()
        
        # Populate the instance matrix with jobs and its (machine, time) tuples
        # instance_matrix[job][op][0 is machine, 1 is time] = (machine, time)
        for job in loaded_instance.jobs:
            for op in range(job.get_operation_count()):
                self.original_instance_matrix[job.id][op] = job.get_operation_as_tuple(op) # (machine, operation time)

        self.max_time_jobs = max(self.original_jobs_length)

        '''
        instance_file = open(instance_path, 'r')
        line_str = instance_file.readline()
        line_cnt = 1
        while line_str:
            split_data = line_str.split()
            if line_cnt == 1:
                self.jobs, self.machines = int(split_data[0]), int(split_data[1])
                # matrix which stores tuples of (machine, length of the job)
                self.instance_matrix = np.zeros((self.jobs, self.machines), dtype=(np.int_, 2))
                # contains all the time to complete jobs
                self.jobs_length = np.zeros(self.jobs, dtype=np.int_)
            else:
                # couple (machine, time)
                assert len(split_data) % 2 == 0 # check if the line is even and contains only (machine, time) couples
                # each jobs must pass a number of operation equal to the number of machines
                assert len(split_data) / 2 == self.machines # check if the number of operation is equal to the number of machines. Each Job has to has the same number of operations??
                i = 0
                # we get the actual jobs
                job_nb = line_cnt - 2 # -2 because we don't count the first line and the line_cnt starts at 1
                while i < len(split_data):
                    machine, time = int(split_data[i]), int(split_data[i + 1]) # (machine, time) tuples
                    self.instance_matrix[job_nb][i // 2] = (machine, time) # // = Floor division operation. 15 // 7 = 2. Rounds the result down to the nearest whole number
                    self.max_time_op = max(self.max_time_op, time) # Check if the operation time is the max operation time
                    self.jobs_length[job_nb] += time
                    self.sum_op += time
                    i += 2
            line_str = instance_file.readline()
            line_cnt += 1
        instance_file.close()
        '''


        
        # check the parsed data are correct
        assert self.max_time_op > 0
        assert self.max_time_jobs > 0
        assert self.jobs > 0
        assert self.machines > 1, 'We need at least 2 machines'
        assert self.original_instance_matrix is not None

        # allocate a job + one to wait
        self.action_space = gym.spaces.Discrete(self.jobs + 1)
        # used for plotting
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]
        '''
        matrix with the following attributes for each job:
            -Legal job
            -Left over time on the current op
            -Current operation %
            -Total left over time
            -When next machine available
            -Time since IDLE: 0 if not available, time otherwise
            -Total IDLE time in the schedule
        '''
        self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs + 1,)),
            "real_obs": gym.spaces.Box(low=0.0, high=1.0, shape=(self.jobs, 7), dtype=np.float),
        })

    def _get_current_state_representation(self):
        #####################################
        # Check for permutation or transposition
        #####################################
        if self.permutation_mode is not None:
            permuted_state = self.state.copy()
            permuted_state, _ = PermutationHandler.permute(permuted_state, self.perm_indices)
            permuted_state[:,0] = self.get_legal_actions()[:-1]
            
            return {
                "real_obs": permuted_state,
                "action_mask": self.get_legal_actions(),
            }
        else:
            state = self.state.copy()
            state[:, 0] = self.get_legal_actions()[:-1]
            return {
                "real_obs": state,
                "action_mask": self.get_legal_actions(),
            }

    def action_masks(self):
        '''
        Just a convenience method, so that you
        do not have to use an ActionMasker later on
        '''
        return self.get_legal_actions

    def get_legal_actions(self):
        #####################################
        # Check for permutation or transposition
        #####################################
        if self.permutation_mode is not None:
            permuted_legal_actions = None
            job_action_mask = np.copy(self.legal_actions[:-1]) # mask without no-op
            permuted_job_action_mask, _ = PermutationHandler.permute(job_action_mask, self.perm_indices)    
            permuted_legal_action = np.append(permuted_job_action_mask, self.legal_actions[-1]).astype(self.legal_actions.dtype) # Add the no-op

            return np.asarray(permuted_legal_action)
        else:
            return np.asarray(np.copy(self.legal_actions))

    def _enable_permutation_mode(self):
        # Just setting self.perm_indices for later use
        if self.permutation_mode == "random":
            # Case: "random" - permute the instance matrix at random
            _, self.perm_indices = PermutationHandler.permute(self.original_instance_matrix)
            self.instance_matrix = np.copy(self.original_instance_matrix)
            #self.jobs_length = PermutationHandler.permute(self.original_instance_matrix)
        elif self.permutation_mode.find("transpose") != -1:
            # Case: "transpose" - permute the instance matrix by transposing it with n_swaps
            n_swaps = int(self.permutation_mode.split("=")[1])
            _, self.perm_indices = PermutationHandler.transpose(self.original_instance_matrix, n_swaps)
            self.instance_matrix = np.copy(self.original_instance_matrix)
    
    def caller_name(self,skip=2):
        """Get a name of a caller in the format module.class.method

        `skip` specifies how many levels of stack to skip while getting caller
        name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

        An empty string is returned if skipped levels exceed stack height
        """
        stack = inspect.stack()
        start = 0 + skip
        if len(stack) < start + 1:
            return ''
        parentframe = stack[start][0]    

        name = []
        module = inspect.getmodule(parentframe)
        # `modname` can be None when frame is executed directly in console
        # TODO(techtonik): consider using __main__
        if module:
            name.append(module.__name__)
        # detect classname
        if 'self' in parentframe.f_locals:
            # I don't know any way to detect call from the object method
            # XXX: there seems to be no way to detect static method call - it will
            #      be just a function call
            name.append(parentframe.f_locals['self'].__class__.__name__)
        codename = parentframe.f_code.co_name
        if codename != '<module>':  # top level usually
            name.append( codename ) # function or a method

        ## Avoid circular refs and frame leaks
        #  https://docs.python.org/2.7/library/inspect.html#the-interpreter-stack
        del parentframe, stack

        return ".".join(name)

    def reset(self):
        print("Resetting the environment...")
        #print(self.caller_name())
        #####################################
        # Check for permutation or transposition
        #####################################
        if self.permutation_mode is not None:
            print(f"Permutation mode is enabled: {self.permutation_mode}")
            self._enable_permutation_mode()
            #self.instance_matrix = np.copy(self.original_instance_matrix)
        else:
            print(f"Permutation mode is disabled: {self.permutation_mode}")
            self.instance_matrix = np.copy(self.original_instance_matrix)
            self.jobs_length = np.copy(self.original_jobs_length)

        self.current_time_step = 0
        self.next_time_step = list()
        self.next_jobs = list()
        self.nb_legal_actions = self.jobs
        self.nb_machine_legal = 0
        # represent all the legal actions
        self.legal_actions = np.ones(self.jobs + 1, dtype=np.bool)
        self.legal_actions[self.jobs] = False
        # used to represent the solution
        self.solution = np.full((self.jobs, self.machines), -1, dtype=np.int_)
        self.time_until_available_machine = np.zeros(self.machines, dtype=np.int_)
        self.time_until_finish_current_op_jobs = np.zeros(self.jobs, dtype=np.int_)
        self.todo_time_step_job = np.zeros(self.jobs, dtype=np.int_)
        self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=np.int_)
        self.needed_machine_jobs = np.zeros(self.jobs, dtype=np.int_)
        self.total_idle_time_jobs = np.zeros(self.jobs, dtype=np.int_)
        self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=np.int_)
        self.illegal_actions = np.zeros((self.machines, self.jobs), dtype=np.bool)
        self.action_illegal_no_op = np.zeros(self.jobs, dtype=np.bool)
        self.machine_legal = np.zeros(self.machines, dtype=np.bool)
        for job in range(self.jobs):
            needed_machine = self.instance_matrix[job][0][0]
            self.needed_machine_jobs[job] = needed_machine
            if not self.machine_legal[needed_machine]:
                self.machine_legal[needed_machine] = True
                self.nb_machine_legal += 1
        self.state = np.zeros((self.jobs, 7), dtype=np.float)
        return self._get_current_state_representation()

    def _prioritization_non_final(self):
        if self.nb_machine_legal >= 1:
            for machine in range(self.machines):
                if self.machine_legal[machine]:
                    final_job = list()
                    non_final_job = list()
                    min_non_final = float('inf')
                    for job in range(self.jobs):
                        if self.needed_machine_jobs[job] == machine and self.legal_actions[job]:
                            if self.todo_time_step_job[job] == (self.machines - 1):
                                final_job.append(job)
                            else:
                                current_time_step_non_final = self.todo_time_step_job[job]
                                time_needed_legal = self.instance_matrix[job][current_time_step_non_final][1]
                                machine_needed_nextstep = self.instance_matrix[job][current_time_step_non_final + 1][0]
                                if self.time_until_available_machine[machine_needed_nextstep] == 0:
                                    min_non_final = min(min_non_final, time_needed_legal)
                                    non_final_job.append(job)
                    if len(non_final_job) > 0:
                        for job in final_job:
                            current_time_step_final = self.todo_time_step_job[job]
                            time_needed_legal = self.instance_matrix[job][current_time_step_final][1]
                            if time_needed_legal > min_non_final:
                                self.legal_actions[job] = False
                                self.nb_legal_actions -= 1

    def _check_no_op(self):
        self.legal_actions[self.jobs] = False
        if len(self.next_time_step) > 0 and self.nb_machine_legal <= 3 and self.nb_legal_actions <= 4:
            machine_next = set()
            next_time_step = self.next_time_step[0]
            max_horizon = self.current_time_step
            max_horizon_machine = [self.current_time_step + self.max_time_op for _ in range(self.machines)]
            for job in range(self.jobs):
                if self.legal_actions[job]:
                    time_step = self.todo_time_step_job[job]
                    machine_needed = self.instance_matrix[job][time_step][0]
                    time_needed = self.instance_matrix[job][time_step][1]
                    end_job = self.current_time_step + time_needed
                    if end_job < next_time_step:
                        return
                    max_horizon_machine[machine_needed] = min(max_horizon_machine[machine_needed], end_job)
                    max_horizon = max(max_horizon, max_horizon_machine[machine_needed])
            for job in range(self.jobs):
                if not self.legal_actions[job]:
                    if self.time_until_finish_current_op_jobs[job] > 0 and \
                            self.todo_time_step_job[job] + 1 < self.machines:
                        time_step = self.todo_time_step_job[job] + 1
                        time_needed = self.current_time_step + self.time_until_finish_current_op_jobs[job]
                        while time_step < self.machines - 1 and max_horizon > time_needed:
                            machine_needed = self.instance_matrix[job][time_step][0]
                            if max_horizon_machine[machine_needed] > time_needed and self.machine_legal[machine_needed]:
                                machine_next.add(machine_needed)
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.jobs] = True
                                    return
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1
                    elif not self.action_illegal_no_op[job] and self.todo_time_step_job[job] < self.machines:
                        time_step = self.todo_time_step_job[job]
                        machine_needed = self.instance_matrix[job][time_step][0]
                        time_needed = self.current_time_step + self.time_until_available_machine[machine_needed]
                        while time_step < self.machines - 1 and max_horizon > time_needed:
                            machine_needed = self.instance_matrix[job][time_step][0]
                            if max_horizon_machine[machine_needed] > time_needed and self.machine_legal[machine_needed]:
                                machine_next.add(machine_needed)
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.jobs] = True
                                    return
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1

    def step(self, action: int):
        reward = 0.0
        original_action = action
        #####################################
        # Check for permutation or transposition
        #####################################
        if self.permutation_mode is not None:
            
            action = PermutationHandler.get_inverse_permuted_action_index(original_action, self.perm_indices)
            #print(f"Taking action {original_action} from the agent and permute it to {action}....")

        # Taking the last job? Is that the no-op?
        if action == self.jobs:

            #print("The action is mapping to a job in the step function...")
            self.nb_machine_legal = 0
            self.nb_legal_actions = 0
            for job in range(self.jobs):
                if self.legal_actions[job]:
                    self.legal_actions[job] = False
                    needed_machine = self.needed_machine_jobs[job]
                    self.machine_legal[needed_machine] = False
                    self.illegal_actions[needed_machine][job] = True
                    self.action_illegal_no_op[job] = True
            while self.nb_machine_legal == 0:
                reward -= self._increase_time_step()
            scaled_reward = self._reward_scaler(reward)
            self._prioritization_non_final()
            self._check_no_op()
            return self._get_current_state_representation(), scaled_reward, self._is_done(), {"makespan": self.current_time_step}
        else:
            #print("The action is not mapping to a job in the step function...")
            current_time_step_job = self.todo_time_step_job[action]
            machine_needed = self.needed_machine_jobs[action]
            time_needed = self.instance_matrix[action][current_time_step_job][1]
            reward += time_needed
            self.time_until_available_machine[machine_needed] = time_needed
            self.time_until_finish_current_op_jobs[action] = time_needed
            self.state[action][1] = time_needed / self.max_time_op
            to_add_time_step = self.current_time_step + time_needed
            if to_add_time_step not in self.next_time_step:
                index = bisect.bisect_left(self.next_time_step, to_add_time_step)
                self.next_time_step.insert(index, to_add_time_step)
                self.next_jobs.insert(index, action)
            self.solution[action][current_time_step_job] = self.current_time_step
            for job in range(self.jobs):
                if self.needed_machine_jobs[job] == machine_needed and self.legal_actions[job]:
                    self.legal_actions[job] = False
                    self.nb_legal_actions -= 1
            self.nb_machine_legal -= 1
            self.machine_legal[machine_needed] = False
            for job in range(self.jobs):
                if self.illegal_actions[machine_needed][job]:
                    self.action_illegal_no_op[job] = False
                    self.illegal_actions[machine_needed][job] = False
            # if we can't allocate new job in the current timestep, we pass to the next one
            while self.nb_machine_legal == 0 and len(self.next_time_step) > 0:
                reward -= self._increase_time_step()
            self._prioritization_non_final()
            self._check_no_op()
            # we then need to scale the reward
            scaled_reward = self._reward_scaler(reward)
            
            return self._get_current_state_representation(), scaled_reward, self._is_done(), {"makespan": self.current_time_step}

    def _reward_scaler(self, reward):
        return reward / self.max_time_op

    def _increase_time_step(self):
        """
        The heart of the logic his here, we need to increase every counter when we have a nope action called
        and return the time elapsed
        :return: time elapsed
        """
        hole_planning = 0
        next_time_step_to_pick = self.next_time_step.pop(0)
        self.next_jobs.pop(0)
        difference = next_time_step_to_pick - self.current_time_step
        self.current_time_step = next_time_step_to_pick
        for job in range(self.jobs):
            was_left_time = self.time_until_finish_current_op_jobs[job]
            if was_left_time > 0:
                performed_op_job = min(difference, was_left_time)
                self.time_until_finish_current_op_jobs[job] = max(0, self.time_until_finish_current_op_jobs[
                    job] - difference)
                self.state[job][1] = self.time_until_finish_current_op_jobs[job] / self.max_time_op
                self.total_perform_op_time_jobs[job] += performed_op_job
                self.state[job][3] = self.total_perform_op_time_jobs[job] / self.max_time_jobs
                if self.time_until_finish_current_op_jobs[job] == 0:
                    self.total_idle_time_jobs[job] += (difference - was_left_time)
                    self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
                    self.idle_time_jobs_last_op[job] = (difference - was_left_time)
                    self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
                    self.todo_time_step_job[job] += 1
                    self.state[job][2] = self.todo_time_step_job[job] / self.machines
                    if self.todo_time_step_job[job] < self.machines:
                        self.needed_machine_jobs[job] = self.instance_matrix[job][self.todo_time_step_job[job]][0]
                        self.state[job][4] = max(0, self.time_until_available_machine[
                            self.needed_machine_jobs[job]] - difference) / self.max_time_op
                    else:
                        self.needed_machine_jobs[job] = -1
                        # this allow to have 1 is job is over (not 0 because, 0 strongly indicate that the job is a
                        # good candidate)
                        self.state[job][4] = 1.0
                        if self.legal_actions[job]:
                            self.legal_actions[job] = False
                            self.nb_legal_actions -= 1
            elif self.todo_time_step_job[job] < self.machines:
                self.total_idle_time_jobs[job] += difference
                self.idle_time_jobs_last_op[job] += difference
                self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
                self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
        for machine in range(self.machines):
            if self.time_until_available_machine[machine] < difference:
                empty = difference - self.time_until_available_machine[machine]
                hole_planning += empty
            self.time_until_available_machine[machine] = max(0, self.time_until_available_machine[
                machine] - difference)
            if self.time_until_available_machine[machine] == 0:
                for job in range(self.jobs):
                    if self.needed_machine_jobs[job] == machine and not self.legal_actions[job] and not \
                            self.illegal_actions[machine][job]:
                        self.legal_actions[job] = True
                        self.nb_legal_actions += 1
                        if not self.machine_legal[machine]:
                            self.machine_legal[machine] = True
                            self.nb_machine_legal += 1
        return hole_planning

    def _is_done(self):
        if self.nb_legal_actions == 0:
            self.last_time_step = self.current_time_step
            return True
        return False

    def render(self, mode='human'):
        df = []
        for job in range(self.jobs):
            i = 0
            while i < self.machines and self.solution[job][i] != -1:
                dict_op = dict()
                dict_op["Task"] = 'Job {}'.format(job)
                start_sec = self.start_timestamp + self.solution[job][i]
                finish_sec = start_sec + self.instance_matrix[job][i][1]
                dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                dict_op["Resource"] = "Machine {}".format(self.instance_matrix[job][i][0])
                df.append(dict_op)
                i += 1
        fig = None
        if len(df) > 0:
            df = pd.DataFrame(df)
            fig = ff.create_gantt(df, index_col='Resource', colors=self.colors, show_colorbar=True,
                                  group_tasks=True)
            fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
        return fig


'''
This function is here just for debugging purposes
'''
if __name__ == "__main__":
    env = JssEnv()