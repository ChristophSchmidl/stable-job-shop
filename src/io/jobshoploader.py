import os
from src.models import Job, JobShopInstance

class JobShopLoader:
    @staticmethod
    def load_jobshop(filename, input_format="JSSP"):
        '''
        Follows the JSSP format
        '''
        if input_format == "JSSP":
            return JobShopLoader._load_jssp_jobshop(filename)
        else:
            raise NotImplementedError()

    def load_jssp_instance_as_list(filename):
        with open(filename) as f:
            lines = f.readlines()
    
        first_line = lines[0].split()
    
        # Number of jobs
        job_count = int(first_line[0])
        # Number of machines
        machine_count = int(first_line[1])
    
        stripped_list = [] # removed new line chars \n
    
        for line in lines:
            stripped_list.append(line.strip())

        return job_count, machine_count, stripped_list[1:]


    @staticmethod
    def _load_jssp_jobshop(filename):
        with open(filename) as f:
            lines = f.readlines()
        
        first_line = lines[0].split()
      
        # Name of instance
        name = os.path.split(filename)[-1]
        # Number of jobs
        job_count = int(first_line[0])
        # Number of machines
        machine_count = int(first_line[1])

        # Create a nested list of all operations
        #all_operations = [ [] for _ in range(job_count) ]


        instance = JobShopInstance(filename, name, job_count, machine_count)

        for job_id, line in enumerate(lines[1:]):
            machine_duration_array = line.split()

            # Create job object with 0 operations and job_id
            new_job = Job(id=job_id)

            it = iter(machine_duration_array)  
            for machine_id, duration in list(zip(it, it)):
                new_job.add_operation(int(machine_id), int(duration))

            instance.add_job(new_job)

        ###################################
        #       Validity check?
        ###################################
        #assert len(split_data) % 2 == 0 # check if the line is even and contains only (machine, time) couples
        # each jobs must pass a number of operation equal to the number of machines
        #assert len(split_data) / 2 == self.machines # check if the number of operation is equal to the number of machines. Each Job has to has the same number of operations??

        return instance

    




        '''
        jobs = {}
        for job_id, line in enumerate(lines[1:]):
            machine_duration_array = line.split()
            new_job = Job(Id=job_id, r=[], p=[])
            it = iter(machine_duration_array)  
            for machine_id, duration in list(zip(it, it)):
                new_job.r.append(int(machine_id))
                new_job.p.append(int(duration))
            
            jobs[job_id] = new_job
        return jobs
        '''