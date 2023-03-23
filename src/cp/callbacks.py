from ortools.sat.python import cp_model 
import wandb
import time
import collections
from src.logger import get_logger

##############################################################################################
#
#   The following solutioncallbacks are alrady standard.
#   They are just here as inspiration :)
#
#   See also: https://github.com/google/or-tools/blob/stable/ortools/sat/python/cp_model.py
##############################################################################################

class ObjectiveSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Display the objective value and time of intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__start_time = time.time()

    def on_solution_callback(self):
        """Called on each new solution."""
        current_time = time.time()
        obj = self.ObjectiveValue()
        print('Solution %i, time = %0.2f s, objective = %i' %
              (self.__solution_count, current_time - self.__start_time, obj))
        self.__solution_count += 1

    def solution_count(self):
        """Returns the number of solutions found."""
        return self.__solution_count


class VarArrayAndObjectiveSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions (objective, variable values, time)."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__start_time = time.time()

    def on_solution_callback(self):
        """Called on each new solution."""
        current_time = time.time()
        obj = self.ObjectiveValue()
        print('Solution %i, time = %0.2f s, objective = %i' %
              (self.__solution_count, current_time - self.__start_time, obj))
        for v in self.__variables:
            print('  %s = %i' % (v, self.Value(v)), end=' ')
        print()
        self.__solution_count += 1

    def solution_count(self):
        """Returns the number of solutions found."""
        return self.__solution_count


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions (variable values, time)."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__start_time = time.time()

    def on_solution_callback(self):
        """Called on each new solution."""
        current_time = time.time()
        print('Solution %i, time = %0.2f s' %
              (self.__solution_count, current_time - self.__start_time))
        for v in self.__variables:
            print('  %s = %i' % (v, self.Value(v)), end=' ')
        print()
        self.__solution_count += 1

    def solution_count(self):
        """Returns the number of solutions found."""
        return self.__solution_count


##############################################################################################
#
#                                   Custom Callbacks
#
##############################################################################################


class WandbOptimalSolutionCallback(cp_model.CpSolverSolutionCallback):
    """Display the objective value and time of intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__start_time = time.time()
        wandb.log({'time': 0, 'solution_count': self.__solution_count})

    def on_solution_callback(self):
        """Called on each new solution."""
        current_time = time.time()
        obj = self.ObjectiveValue()
        self.__solution_count += 1
        wandb.log({'time': current_time - self.__start_time, 'make_span': obj, 'solution_count': self.__solution_count})

    def solution_count(self):
        """Returns the number of solutions found."""
        return self.__solution_count

class OptimalSolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.num_solutions = 0
        self.logger = get_logger()
    
    def on_solution_callback(self):
        self.num_solutions += 1
        self.logger.info(f"Executing on_solution_callback with Response status: {self.Response().status}")
        self.logger.info(f"Proposed feasible solution: {self.Response().solution}")

        if self.Response().status == cp_model.OPTIMAL:
            self.logger.info(f'Found {self.num_solutions} optimal solution(s) with makespan {self.ObjectiveValue()}.')
            self.logger.info(f"Found solution: {self.Response().solution}")
            self.StopSearch()


class JobShopVarArrayAndObjectiveSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions (objective, variable values, time)."""

    def __init__(self, jobs_data, assigned_task_type, all_tasks, all_machines):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__jobs_data = jobs_data
        self.__assigned_task_type = assigned_task_type
        self.__all_tasks = all_tasks
        self.__all_machines = all_machines
        self.__solution_count = 0
        self.__start_time = time.time()

    def _get_assigned_jobs(self):
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(self.__jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    self.__assigned_task_type(start = self.Value(
                        self.__all_tasks[job_id, task_id].start),
                                       job=job_id,
                                       index=task_id,
                                       duration=task[1]))
        return assigned_jobs

    def _print_per_machine_solution(self, assigned_jobs):
        # Create per machine output lines.
        output = ''
        for i, machine in enumerate(self.__all_machines):
            #machine_tasks = []
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '
            #ax = self.axs[i]


            for j, assigned_task in enumerate(assigned_jobs[machine]):

                name = '(%i,%i)' % (assigned_task.job,
                                           assigned_task.index)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-15s' % name
                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = '[%i,%i]' % (start, start + duration)
                #machine_tasks.append((assigned_task.job, assigned_task.index, start, start + duration))
                # Add spaces to output to align columns.
                sol_line += '%-15s' % sol_tmp
                #self.solution_arr.append(dict(Job=f"Job {assigned_task.job}", Start=start, Finish=start + duration, Machine=f"Machine {machine}"))

                #machine_tasks.append((assigned_task.job, assigned_task.index, start, start + duration))
            #self.solution_dict[machine] = machine_tasks
            #ax.set_title(f"Machine {machine}")
            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print(f'Optimal Schedule Length: {self.ObjectiveValue()}')
        print(output)

    def on_solution_callback(self):
        """Called on each new solution."""
        current_time = time.time()
        obj = self.ObjectiveValue()
        print(f"Solution {self.__solution_count}, time = {current_time - self.__start_time}, Objective = {obj}")

        assigned_jobs = self._get_assigned_jobs()
        self._print_per_machine_solution(assigned_jobs)

        self.__solution_count += 1

    def solution_count(self):
        """Returns the number of solutions found."""
        return self.__solution_count