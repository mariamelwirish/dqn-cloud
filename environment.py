from collections import deque
import numpy as np


class Task:
    _counter = 0

    @classmethod
    def reset_counter(cls):
        cls._counter = 0

    def __init__(self, arrival_time, cpu_demand, memory_demand, deadline, is_io):
        # Identity
        Task._counter += 1
        self.task_id = Task._counter

        # Given Task Parameters
        self.arrival_time = arrival_time
        self.cpu_demand = cpu_demand
        self.memory_demand = memory_demand
        self.deadline = deadline
        self.is_io = is_io

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
        self.queue = deque() # a queue of Task objects
        self.running_task = None # This should point to the task object currently executing on the VM. None if idle


class CloudEnvironment:
    ''' Simulation Parameters '''
    # VM
    MIN_CPU_SPEED = 500 # MIPS
    MAX_CPU_SPEED = 2000 # MIPS
    MIN_MEMORY_CAPACITY = 1024 # MBs
    MAX_MEMORY_CAPACITY = 65536 # MBs

    # Task 
    MIN_JOB_LENGTH = 100 # MIPS
    MAX_JOB_LENGTH = 1500 # MIPS
    MIN_MEMORY_REQUEST = 100 # MBs
    MAX_MEMORY_REQUEST = 1000 # MBs

    # VM tiers derived from Google Cluster Trace 2019 normalized capacity structure
    # Reference machine: 1000 MIPS, 32 GB RAM
    VM_TIERS = [
        {'cpu_speed': 500,  'memory_capacity': 16384,  'count': 3},  # small  (0.5x NCU)
        {'cpu_speed': 1000, 'memory_capacity': 32768,  'count': 4},  # medium (1.0x NCU)
        {'cpu_speed': 2000, 'memory_capacity': 65536,  'count': 3},  # large  (2.0x NCU)
    ]

    ALPHA = 0.5

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

        self.cascade_misses = 0

        

    # def _create_vms(self):
    #     vms = []
    #     for _ in range(self.n_vms):
    #         cpu_speed = self.seed.uniform(CloudEnvironment.MIN_CPU_SPEED, CloudEnvironment.MAX_CPU_SPEED) # CPU speed in MIPS
    #         memory_capacity = self.seed.uniform(CloudEnvironment.MIN_MEMORY_CAPACITY, CloudEnvironment.MAX_MEMORY_CAPACITY) # Memory capacity in MBs
    #         vms.append(VM(cpu_speed, memory_capacity))
    #     return vms

    def _create_vms(self):
        vms = []
        for tier in CloudEnvironment.VM_TIERS:
            for _ in range(tier['count']):
                vms.append(VM(tier['cpu_speed'], tier['memory_capacity']))
        self.seed.shuffle(vms)  # randomize order so tier pattern isn't positional
        # for vm in vms: print(vm.cpu_speed, vm.memory_capacity)
        return vms


    
    # def _generate_tasks(self):
    #     tasks = []
    #     arrival_time = 0.0
    #     avg_processing_time = (CloudEnvironment.MIN_CPU_SPEED + CloudEnvironment.MAX_CPU_SPEED) / 2
    #     job_length = np.clip(self.seed.lognormal(mean=5.5, sigma=0.8), 50, 1500)


    #     for _ in range(self.n_tasks):
    #         # Arrival time
    #         inter_arrival_time = self.seed.exponential(1 / self.arrival_rate)
    #         arrival_time += inter_arrival_time

    #         # Resources Demand
    #         job_length = self.seed.uniform(CloudEnvironment.MIN_JOB_LENGTH, CloudEnvironment.MAX_JOB_LENGTH) # Job Length in MIPS
    #         memory_demand = self.seed.uniform(CloudEnvironment.MIN_MEMORY_REQUEST, CloudEnvironment.MAX_MEMORY_REQUEST) # RAM demand in MBs

    #         # Deadline 
    #         # Duration = (ci / Cj) * Uniform Random Multiplier
    #         expected_execution_time = job_length / avg_processing_time
    #         multiplier = self.seed.uniform(2.0, 4.0) 
    #         deadline = arrival_time + expected_execution_time * multiplier

    #         is_io = self.seed.random() < 0.5
    #         # Create Task and add to list
    #         tasks.append(Task(arrival_time, job_length, memory_demand, deadline, is_io))

    #     for task in tasks[:10]:
    #         print(f"{task.cpu_demand:.1f}")
    #     return tasks

    def _generate_tasks(self):
        tasks = []
        arrival_time = 0.0
        avg_processing_time = (CloudEnvironment.MIN_CPU_SPEED + CloudEnvironment.MAX_CPU_SPEED) / 2

        for _ in range(self.n_tasks):
            inter_arrival_time = self.seed.exponential(1 / self.arrival_rate)
            arrival_time += inter_arrival_time

            job_length = np.clip(self.seed.lognormal(mean=5.5, sigma=0.8), 50, 1500)  # ← inside loop
            memory_demand = self.seed.uniform(CloudEnvironment.MIN_MEMORY_REQUEST, CloudEnvironment.MAX_MEMORY_REQUEST)

            expected_execution_time = job_length / avg_processing_time
            multiplier = self.seed.uniform(2.0, 4.0)
            deadline = arrival_time + expected_execution_time * multiplier

            is_io = self.seed.random() < 0.5
            tasks.append(Task(arrival_time, job_length, memory_demand, deadline, is_io))

        return tasks
    
    def _check_preemption(self, task, vm):
        # 1. Urgency: arriving task has tighter deadline than the running one. 
        urgency = task.deadline < vm.running_task.deadline

        # 2. Necessity: waiting in queue would cause the task to miss its deadline.
        necessity = vm.available_time + task.execution_time > task.deadline

        # 3. Safety: preempting the running task would not cause it to miss its deadline.
        safety = (self.current_time + task.execution_time + vm.running_task.remaining_time) <= vm.running_task.deadline 

        # 4. Cascade 
        cascade = not any(t.num_preemptions > 0 for t in vm.queue)

        return urgency and necessity and safety and cascade


    def _preempt(self, task, vm):   
        # print(f"PREEMPTION: task {task.task_id} (deadline={task.deadline:.3f}) preempts task {preempted_task.task_id} (deadline={preempted_task.deadline:.3f}) at t={self.current_time:.3f}")        # Update the preempted task's parameters
        preempted_task = vm.running_task
        preempted_task.num_preemptions += 1
        preempted_task.queue_entry_times.append(self.current_time)

        # Preempt the running task and assign the new task to the VM
        vm.running_task = task
        task.service_start_times.append(self.current_time)
        vm.available_time += task.execution_time

        # Add the preempted task in the front of the queue
        vm.queue.appendleft(preempted_task)

        self._recalculate_cascade(vm)


    def _recalculate_cascade(self, vm):
        estimated_start = self.current_time + vm.running_task.execution_time
        for task in vm.queue:
            exe = task.remaining_time if task.num_preemptions > 0 else task.execution_time
            estimated_completion = estimated_start + exe
            if estimated_completion > task.deadline:
                task.success = 0
                self.cascade_misses += 1
            estimated_start = estimated_completion
            task.waiting_time += vm.running_task.execution_time
            task.response_time = task.waiting_time + exe
        

    def assign_task(self, task, vm):
        self.cascade_misses = 0
        if task.is_io:
            task.execution_time = 2 * task.cpu_demand / vm.cpu_speed 
        else:
            task.execution_time = task.cpu_demand / vm.cpu_speed 
        task.remaining_time = task.execution_time
        task.queue_entry_times.append(task.arrival_time)

        # Check if running task has completed — transition VM state
        if vm.running_task is not None and vm.available_time <= self.current_time:
            vm.running_task = None
            if vm.queue:
                next_task = vm.queue.popleft()
                next_task.service_start_times.append(self.current_time)
                vm.running_task = next_task

        if vm.running_task is not None:
            # Update remaining time of currently running task
            vm.running_task.remaining_time = vm.running_task.execution_time - (self.current_time - vm.running_task.service_start_times[-1])
            
            if self._check_preemption(task, vm):
                self._preempt(task, vm)
                task.waiting_time = 0.0
            else:
                vm.queue.append(task)
                task.waiting_time = max(0, vm.available_time - task.arrival_time)
                vm.available_time += task.execution_time
        else:
            # VM is idle — assign directly
            task.service_start_times.append(self.current_time)
            vm.running_task = task
            vm.available_time = self.current_time + task.execution_time
            task.waiting_time = 0.0

        task.response_time = task.waiting_time + task.execution_time
        task.success = 1 if task.arrival_time + task.response_time <= task.deadline else 0

    def get_feasible_vms(self, task):
        return [vm for vm in self.vms if vm.memory_capacity >= task.memory_demand]
    
    def get_state(self, task):
        state = []

        # Task features — normalized
        deadline_remaining = (task.deadline - self.current_time) / task.deadline
        task_features = [
            task.cpu_demand / CloudEnvironment.MAX_JOB_LENGTH,
            task.memory_demand / CloudEnvironment.MAX_MEMORY_REQUEST,
            max(0, deadline_remaining)
        ]
        state.extend(task_features)

        # VM features — normalized
        max_available_time = max(vm.available_time for vm in self.vms) + 1e-9
        for vm in self.vms:
            preemption_flag = 1 if vm.running_task and self._check_preemption(task, vm) else 0
            vm_features = [
                vm.available_time / max_available_time,
                vm.cpu_speed / CloudEnvironment.MAX_CPU_SPEED,
                vm.memory_capacity / CloudEnvironment.MAX_MEMORY_CAPACITY,
                preemption_flag
            ]
            state.extend(vm_features)

        return state
    
    def compute_reward(self, task):
        margin = (task.deadline - (task.arrival_time + task.response_time)) / task.deadline
        reward = margin - 0.2 * self.cascade_misses
        return reward
        