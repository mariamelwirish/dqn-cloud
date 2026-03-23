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
        return self.v_cost * job.exe_time

    def calculate_success(self, job: Job):
        if job.response_time < job.qos:
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
    def __init__(self, n_vms, n_jobs, io_vm_ratio, io_job_ratio, mean_job_arrival_rate, mean_job_length, variance_job_length, ):
        self.n_vms = n_vms
        self.n_jobs = n_jobs
        self.io_vm_ratio = io_vm_ratio
        self.io_job_ratio = io_job_ratio
        self.mean_job_arrival_rate = mean_job_arrival_rate
        self.mean_job_length = mean_job_length
        self.variance_job_length = variance_job_length

        self.vms = self.create_vms()
        self.jobs = self.generate_jobs()
        self.current_time = 0
        self.completed_jobs = []

    def create_vms(self) -> list[VM]:
        vms = []
        np.random.seed(42)
        # create exactly the right number of each type
        n_io_vms = int(self.n_vms * self.io_vm_ratio)
        types = ['I/O'] * n_io_vms + ['CPU'] * (self.n_vms - n_io_vms)
        np.random.shuffle(types)  # shuffle so they're not all I/O first

        for i in range(self.n_vms):
            vm_type = types[i]
            v_com = np.random.uniform(1000, 3000)
            v_cost = np.random.uniform(4, 5)
            vms.append(VM(i, v_com, vm_type, v_cost))
        
        return vms

    def generate_jobs(self):
        np.random.seed(42)
        jobs = []
        current_time = 0

        # create exactly the right number of each type
        n_io_jobs = int(self.n_jobs * self.io_job_ratio)
        types = ['I/O'] * n_io_jobs + ['CPU'] * (self.n_jobs - n_io_jobs)
        np.random.shuffle(types)  # shuffle so they're not all I/O first

        for i in range(self.n_jobs):
            job_type = types[i]
            inter_arrival = np.random.exponential(1 / self.mean_job_arrival_rate)
            current_time += inter_arrival
            arrival_time = current_time            
            req_com = max(1, np.random.normal(self.mean_job_length, np.sqrt(self.variance_job_length)))
            # qos = np.random.uniform(3, 10) 
            exe_time_estimate = req_com / 2000 # assuming average VM speed
            qos = exe_time_estimate * np.random.uniform(2, 4)  # 10x-50x execution time
            jobs.append(Job(i, arrival_time, req_com, job_type, qos))
        return jobs
    
    def get_state(self, job: Job):
        waiting_times = []
        for vm in self.vms:
            waiting_time = vm.calculate_waiting_time(job)
            waiting_times.append(waiting_time)
        
        return [job.req_com, job.qos] + waiting_times
    
    def calculate_reward(self, job: Job):
        if job.success == 1:
            return (1 / job.response_time) + (1 / job.cost)
        else:
            return 0