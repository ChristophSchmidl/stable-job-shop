from ortools.sat.python import cp_model 
from ortools.sat.python.cp_model import  VarArrayAndObjectiveSolutionPrinter
import numpy as np
import os
import wandb
import collections
from src import config
from src.io.jobshoploader import JobShopLoader
from itertools import cycle
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import chart_studio
import chart_studio.plotly as py
from src.logger import get_logger
from src.cp.callbacks import WandbOptimalSolutionCallback, OptimalSolutionCallback, JobShopVarArrayAndObjectiveSolutionPrinter

class CPJobShopSolver:
    def __init__(self, filename="data/instances/taillard/ta41.txt", logger=None):
        # Load the problem instance
        self.logger = get_logger() if logger is None else logger
        self.jobs_count, self.machines_count, self.jobs_data = self._load_instance(filename)
        # self.jobs_data.shape: (job_count, machine_count, 2)
        '''
        [
            [ # job_id
                [5 94] # machine_id processing_time
                [12 66]
                [4 10]
            ]
        ]
        '''
        self.instance_name = os.path.split(filename)[-1].split(sep=".")[0].upper()

        self.model = None
        self.solver = None
        self.solver_parameters = None

        self.all_machines = None
        self.all_tasks = None
        self.assigned_task_type = None
        self.horizon = None

        # Contains solution dictionary where every key is the index of the machine pointing to an array of tuples
        # with (job_id, task_id, start_time, end_time (start + duration))
        #  Example: {0, [(7,2,84,93)]}
        self.solution_arr = []
        self.solution_found = False

        self.fig = None
        self.axs = None

        self._init_cp_model()

    def _load_instance(self, filename):
        jobs_data = []
        jobs_count = 0
        machines_count = 0

        # Refactor with logging behavior?
        if os.path.isfile(filename):
            self.logger.info(f"Loading instance from file: {filename}")

            with open(filename) as f:
                line_str = f.readline()
                line_count = 1

                while line_str:
                    data = []
                    split_data = line_str.split()
                    if line_count == 1:
                        jobs_count, machines_count = int(split_data[0]), int(split_data[1])
                    else:
                        i = 0
                        while i < len(split_data):
                            machine, time = int(split_data[i]), int(split_data[i+1])
                            data.append((machine, time))
                            i += 2
                        jobs_data.append(data)
                    line_str = f.readline()
                    line_count += 1

        else:
            self.logger.info(f"File not found: {filename}.")
            return

        self.logger.info(f"Successfully loaded instance {filename} with {jobs_count} jobs and {machines_count} machines.")
        return jobs_count, machines_count, np.array(jobs_data)

    def _init_cp_model(self):
        self.logger.info(f"Initializing cp solver...")
        self.model = cp_model.CpModel()

        self.all_machines = range(self.machines_count)

        # Computes the horizon dynamically as the sum of all durations.
        self.horizon = sum(task[1] for job in self.jobs_data for task in job)
        self.logger.info(f"Horizon is: {self.horizon}")

        # Named tuple to store information about created variables.
        task_type = collections.namedtuple('task_type', 'start end interval')
        # Named tuple to manipulate solution information.
        self.assigned_task_type = collections.namedtuple('assigned_task_type',
                                                    'start job index duration')

        # Creates job intervals and add to the corresponding machine lists.
        self.all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        for job_id, job in enumerate(self.jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                duration = task[1]
                suffix = '_%i_%i' % (job_id, task_id)
                start_var = self.model.NewIntVar(0, self.horizon, 'start' + suffix)
                end_var = self.model.NewIntVar(0, self.horizon, 'end' + suffix)
                interval_var = self.model.NewIntervalVar(start_var, duration, end_var,
                                                    'interval' + suffix)
                self.all_tasks[job_id, task_id] = task_type(start=start_var,
                                                       end=end_var,
                                                       interval=interval_var)
                machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints.
        for machine in self.all_machines:
            self.model.AddNoOverlap(machine_to_intervals[machine])

        # Precedences inside a job.
        for job_id, job in enumerate(self.jobs_data):
            for task_id in range(len(job) - 1):
                self.model.Add(self.all_tasks[job_id, task_id +
                                    1].start >= self.all_tasks[job_id, task_id].end)

        # Makespan objective.
        obj_var = self.model.NewIntVar(0, self.horizon, 'makespan')
        self.model.AddMaxEquality(obj_var, [
            self.all_tasks[job_id, len(job) - 1].end
            for job_id, job in enumerate(self.jobs_data)
        ])
        self.model.Minimize(obj_var)

    def _get_assigned_jobs(self, jobs_data, assigned_task_type, all_tasks, solver):
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(start = solver.Value(
                        all_tasks[job_id, task_id].start),
                                       job=job_id,
                                       index=task_id,
                                       duration=task[1]))
        return assigned_jobs

    def _print_per_machine_solution(self, all_machines, assigned_jobs):
        # Create per machine output lines.
        output = ''
        for i, machine in enumerate(all_machines):
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
        print(f'Optimal Schedule Length: {self.solver.ObjectiveValue()}')
        print(output)

    def solve(self, max_time=10.0):
        # Initialize a new run

        if config.USE_WANDB:
            run = wandb.init(
                project=config.WANDB_PROJECT,
                notes="CP solver",
                group="constraint-programming",
                job_type="",
                tags=["cp", "baseline"]    
            )

        self.solver = cp_model.CpSolver()

        if max_time != 0:
            self.solver.parameters.max_time_in_seconds = max_time * 60

        if config.USE_WANDB:
            wandb_callback = WandbOptimalSolutionCallback()
            status = self.solver.Solve(self.model, wandb_callback)
        else:
            self.logger.info("Using JobShopVarArrayAndObjectiveSolutionPrinter")
            
            solution_printer = JobShopVarArrayAndObjectiveSolutionPrinter(
                self.jobs_data, 
                self.assigned_task_type, 
                self.all_tasks, 
                self.all_machines
            )

            status = self.solver.Solve(self.model, solution_printer)

        if status == cp_model.OPTIMAL:
        #if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            if status == cp_model.OPTIMAL:
                self.logger.info(f"Optimal schedule found with makespan {self.solver.ObjectiveValue()}")
                self.solution_found = True
            #elif status == cp_model.FEASIBLE:
            #    print(f"Feasible schedule found with makespan {self.solver.ObjectiveValue()}")
            #    self.solution_found = True

            #assigned_jobs = self._get_assigned_jobs(
            #    self.jobs_data, 
            #    self.assigned_task_type, 
            #    self.all_tasks, 
            #    self.solver)

            #for k, v in assigned_jobs.items():
            #    self.logger.info(f"{k} - {v}")


            #self.logger.info(f"Assigned jobs: {assigned_jobs}")
            
            #self._print_per_machine_solution(self.all_machines, assigned_jobs)
            exit()
          
        else:
            print('No solution found.')

        #return self.solution_dict
        if config.USE_WANDB:
            run.finish()

        return self.solution_arr

    def save_figure(self, file_path, resolution=300):
        self.fig.savefig(file_path, dpi=resolution)
        plt.close(self.fig)

    def create_gantt_chart(self, y_axis="Machine"):
        username = 'cschmidl' # your username
        api_key = 'HHCrbjR60TTnhU4mF7HX' # your api key - go to profile > settings > regenerate key
        chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
        

        '''
        y_axis = "Machine" or "Job"
        '''

        # plt.close()
        if len(self.solution_arr) == 0:
            print("Cannot create Gantt chart. Solution array is empty.")
        
        plt.figure()

        '''
        Dataframe description for solution:
        |Job|Task|Machine|Start|Finish|Duration|

        #self.solution_arr.append(dict(Job=f"Job {assigned_task.job}", Start=start, Finish=start + duration, Machine=f"Machine {machine}"))
        '''



        df = pd.DataFrame(self.solution_arr)
        df['Duration'] = df['Finish'] - df['Start']

        #fig = px.timeline(df, x_start="Start", x_end="Finish", y="Machine", color="Job")
        #fig.update_yaxes(autorange="reversed")
        #fig.layout.xaxis.type = 'linear'

        fig = px.bar(df, 
            base = "Start",
            x = "Duration",
            y = "Machine" if y_axis == "Machine" else "Job",
            color = "Job" if y_axis == "Machine" else "Machine",
            orientation = 'h'
        )

        fig.update_yaxes(autorange="reversed")
        fig.update_layout(hovermode="x unified")

        py.plot(fig, filename = f"ta01_cp_optimal_group_by_{y_axis.lower()}", auto_open=True)
        fig.show()

        
    def show_figure(self):
        '''
        Displays the solution as a gantt chart (Matplotlib hbar)
        '''
        plt.show()

    def print_statistics(self):
        if self.solution_found:
            # Statistics.
            print('\nStatistics')
            print('  - conflicts: %i' % self.solver.NumConflicts())
            print('  - branches : %i' % self.solver.NumBranches())
            print('  - wall time: %f s' % self.solver.WallTime())
        else:
            print(f"Cannot print statistics because there was no solution found.")

    def get_statistics(self):
        if self.solution_found:
            # Statistics.
            return self.solver.NumConflicts(), self.solver.NumBranches(), self.solver.WallTime()
        else:
            print(f"Cannot get statistics because there was no solution found.")

def generate_instance_paths(interval_begin, interval_end):
    return [f'data/instances/taillard/ta{i:02d}.txt' for i in range(interval_begin, interval_end + 1)]



if __name__ == '__main__':
    #INSTANCE_PATH = f"data/instances/taillard/ta41.txt"
    #INSTANCE_NAME =  os.path.split(INSTANCE_PATH)[-1].split(sep=".")[0].upper()
    
    INSTANCE_PATHS = generate_instance_paths(1, 1)

    for instance in INSTANCE_PATHS:

        cp_solver = CPJobShopSolver(filename=instance)
        solution = cp_solver.solve()
        print(solution)
        #cp_solver.create_gantt_chart(y_axis="Machine")
        #cp_solver.show_figure()
        #cp_solver.save_figure("plots/ta01_cp_optimal_schedule.png")

    

    
    



