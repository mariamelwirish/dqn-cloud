import numpy as np

'''
Environment consistes of the following:
    1. Job class.
    2. VM class.
    3. CloudEnvironment class (where simulation happens).
'''

class Job:
    def __init__(self, job_id, arrival_time, req_com, job_type, qos):
        self.job_id = job_id 
        self.arrival_time = arrival_time
        self.req_com = req_com          # number of instructions
        self.job_type = job_type        # CPU or I/O
        self.qos = qos                  # deadline (expected time)

        self.exe_time = 0
        self.wait_time = 0
        self.response_time = 0
        self.cost = 0
        self.success = 0
        self.preempted = False


class VM:
    def __init__(self, vm_id, v_com, vm_type, v_cost):
        self.vm_id = vm_id
        self.v_com = v_com              # computing capacity (instructions per second) 
        self.vm_type = vm_type          # CPU or I/O
        self.v_cost = v_cost            # execution cost per time unit

        self.queue = []                 # list of jobs assigned to this VM
        self.available_time = 0         # when the VM will be available for the next job

    # When a job is assigned to a VM:
        # T(res,job) = T(wait, job) + T(exe,job)
    

    def calculate_waiting_time(self, job: Job):
        if len(self.queue) == 0 or self.available_time <= job.arrival_time:
            return 0
        else:
            return self.available_time - job.arrival_time
    
    def calculate_execution_time(self, job: Job):
        time = job.req_com / self.v_com
        if job.job_type == self.vm_type:
            return time
        else: 
            return 2 * time
        
    def calculate_response_time(self, job: Job):
        return job.wait_time + job.exe_time
    
    def calculate_cost(self, job: Job):
        return 0.1 + job.exe_time * self.v_cost

    def calculate_success(self, job: Job):
        if job.response_time <= job.qos:
            return 1
        else:
            return 0
    
    def assign_job(self, job: Job):
        job.wait_time = self.calculate_waiting_time(job)
        job.exe_time = self.calculate_execution_time(job)
        job.response_time = self.calculate_response_time(job)
        job.cost = self.calculate_cost(job)
        job.success = self.calculate_success(job)

        self.queue.append(job)

        self.available_time = job.arrival_time + job.wait_time + job.exe_time
    

class CloudEnvironment:
    def __init__(self, n_vms, n_jobs, mean_job_arrival_rate, mean_job_length, job_len_std, io_job_ratio, seed,):
        self.n_vms = n_vms
        self.n_jobs = n_jobs
        self.mean_job_arrival_rate = mean_job_arrival_rate
        self.mean_job_length = mean_job_length
        self.job_len_std = job_len_std
        self.io_job_ratio = io_job_ratio
        self.seed = seed

        self.vms = self.create_vms()
        self.jobs = self.generate_jobs()
        self.current_time = 0
        self.completed_jobs = []

    def create_vms(self) -> list[VM]:
        vms = []
        types = [0,0,0,0,0,1,1,1,1,1]
        speed_multipliers = [1,1,1.1,1.1,1.2,1,1,1.1,1.1,1.2]
        costs = [1,1,2,2,4,1,1,2,2,4]

        for i in range(self.n_vms):
            v_com = 1000 * speed_multipliers[i]
            vms.append(VM(i, v_com, types[i], costs[i]))
        
        return vms

    def generate_jobs(self):
        jobs = []
        current_time = 0

        for i in range(self.n_jobs):
            # job type
            if self.seed.uniform() < self.io_job_ratio:
                job_type = 0
            else:
                job_type = 1
            
            # arrival time
            inter_arrival = self.seed.exponential(1 / self.mean_job_arrival_rate)
            current_time += inter_arrival
            arrival_time = current_time
            
            # job length
            req_com = int(self.seed.normal(self.mean_job_length, self.job_len_std))
            
            # qos
            length = req_com / 1000
            qos = 0.25
            
            jobs.append(Job(i, arrival_time, req_com, job_type, qos))
        
        return jobs
    
    def get_state(self, job: Job):
        waiting_times = []
        for vm in self.vms:
            waiting_time = vm.calculate_waiting_time(job)
            waiting_times.append(waiting_time)
        
        return [job.job_type] + waiting_times
    
    def calculate_reward(self, job: Job, vm: VM):
        length = job.req_com / 1000
        cost = 0.1 + job.exe_time * vm.v_cost
        reward = (1 + np.exp(1.5 - cost)) * length / job.response_time
        return reward
            