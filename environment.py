
class Task:
    _counter = 0

    @classmethod
    def reset_counter(cls):
        cls._counter = 0

    def __init__(self, arrival_time, cpu_demand, memory_demand, deadline):
        # Identity
        Task._counter += 1
        self.task_id = Task._counter

        # Given Task Parameters
        self.arrival_time = arrival_time
        self.cpu_demand = cpu_demand
        self.memory_demand = memory_demand
        self.deadline = deadline

        # Derived Task Parameters
        self.execution_time = 0.0
        self.remaining_time = 0.0
        self.num_preemptions = 0
        self.queue_entry_times = []
        self.service_start_times = []
        self.waiting_time = 0.0
        self.response_time = 0.0
        self.success = 0
    

class VM:
    _counter = 0

    @classmethod
    def reset_counter(cls):
        cls._counter = 0

    def __init__(self, cpu_speed, memory_capacity):
        # Identity
        VM._counter += 1
        self.vm_id = VM._counter

        # Given VM Parameters
        self.cpu_speed = cpu_speed
        self.memory_capacity = memory_capacity

        # Derived VM Parameters
        self.available_time = 0.0
        self.queue = [] # a queue of Task objects
        self.running_task = None # This should point to the task object currently executing on the VM. None if idle


class CloudEnvironment:
    ''' Simulation Parameters '''
    # VM
    MIN_CPU_SPEED = 1000 # MIPS
    MAX_CPU_SPEED = 3000 # MIPS
    MIN_MEMORY_CAPACITY = 1024 # MBs
    MAX_MEMORY_CAPACITY = 8192 # MBs

    # Task 
    MIN_JOB_LENGTH = 1000 # MIPS
    MAX_JOB_LENGTH = 20000 # MIPS
    MIN_MEMORY_REQUEST = MIN_MEMORY_CAPACITY / 2 # MBs
    MAX_MEMORY_REQUEST = MAX_MEMORY_CAPACITY / 2 # MBs

    def __init__(self, n_vms, n_tasks, arrival_rate, seed):
        # Given Environment Parameters
        self.n_vms = n_vms
        self.n_tasks = n_tasks
        self.arrival_rate = arrival_rate
        self.seed = seed

        # Environment Management Parameters
        self.current_time = 0.0

        # Environment Core Entities
        self.vms = self._create_vms()
        self.tasks = self._generate_tasks()

        

    def _create_vms(self):
        vms = []
        for _ in range(self.n_vms):
            cpu_speed = self.seed.uniform(CloudEnvironment.MIN_CPU_SPEED, CloudEnvironment.MAX_CPU_SPEED) # CPU speed in MIPS
            memory_capacity = self.seed.uniform(CloudEnvironment.MIN_MEMORY_CAPACITY, CloudEnvironment.MAX_MEMORY_CAPACITY) # Memory capacity in MBs
            vms.append(VM(cpu_speed, memory_capacity))
        return vms

    def _generate_tasks(self):
        tasks = []
        arrival_time = 0.0
        avg_processing_time = (CloudEnvironment.MIN_CPU_SPEED + CloudEnvironment.MAX_CPU_SPEED) / 2


        for _ in range(self.n_tasks):
            # Arrival time
            inter_arrival_time = self.seed.exponential(1 / self.arrival_rate)
            arrival_time += inter_arrival_time

            # Resources Demand
            job_length = self.seed.uniform(CloudEnvironment.MIN_JOB_LENGTH, CloudEnvironment.MAX_JOB_LENGTH) # Job Length in MIPS
            memory_demand = self.seed.uniform(CloudEnvironment.MIN_MEMORY_REQUEST, CloudEnvironment.MAX_MEMORY_REQUEST) # RAM demand in MBs

            # Deadline 
            # Duration = (ci / Cj) * Uniform Random Multiplier
            expected_execution_time = job_length / avg_processing_time
            multiplier = self.seed.uniform(1.5, 3.0) 
            deadline = arrival_time + expected_execution_time * multiplier

            # Create Task and add to list
            tasks.append(Task(arrival_time, job_length, memory_demand, deadline))

        return tasks