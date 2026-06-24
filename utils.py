from operator import index

import numpy as np
from environment import CloudEnvironment
from algorithms.dqn.agent import DQNAgent
from algorithms.random.scheduler import RandomScheduler
from algorithms.round_robin.scheduler import RoundRobinScheduler
from algorithms.earliest.scheduler import EarliestVMScheduler
from environment import CloudEnvironment, Task, VM


class Utils:

    @staticmethod
    def create_environment(seed, arrival_rate):
        """Creates a fresh CloudEnvironment with fixed simulation parameters."""
        return CloudEnvironment(
            n_vms=10,
            n_tasks=8000,
            arrival_rate=arrival_rate,
            seed=seed
        )

    @staticmethod
    def run_dqn(env, seed, n_epochs=5):
        """Runs DQN training loop over multiple epochs. Returns averaged metrics."""
        state_size = 3 + 4 * env.n_vms
        agent = DQNAgent(
            input_size=state_size,
            hidden_size=20,
            output_size=env.n_vms,
            seed=seed
        )

        all_success_rates = []
        all_response_times = []
        all_cascade_rates = []

        for epoch in range(n_epochs):
            # Reset environment for each epoch
            Task.reset_counter()
            VM.reset_counter()
            env.vms = env._create_vms()
            env.tasks = env._generate_tasks()
            env.current_time = 0.0
            env.cascade_misses = 0

            epoch_cascade_misses = 0

            for index, task in enumerate(env.tasks):
                env.current_time = task.arrival_time

                # Get feasible VMs
                feasible_vms = env.get_feasible_vms(task)
                if not feasible_vms:
                    continue

                # Get state and select action
                state = env.get_state(task)
                feasible_indices = [vm.vm_id - 1 for vm in feasible_vms]
                action = agent.select_action(state, feasible_indices)
                selected_vm = env.vms[action]

                # Assign task and compute reward
                env.assign_task(task, selected_vm)
                reward = env.compute_reward(task)
                epoch_cascade_misses += env.cascade_misses

                # Get next state
                if index < len(env.tasks) - 1:
                    next_state = env.get_state(env.tasks[index + 1])
                else:
                    next_state = [0.0] * state_size

                # Store experience and train
                agent.remember(state, action, reward, next_state)
                agent.train()

            # Skip first epoch — agent is still warming up
            if epoch > 0:
                success_rate, avg_response_time, cascade_rate = Utils.calculate_metrics(env.tasks, epoch_cascade_misses)
                all_success_rates.append(success_rate)
                all_response_times.append(avg_response_time)
                all_cascade_rates.append(cascade_rate)

        return (
            sum(all_success_rates) / len(all_success_rates),
            sum(all_response_times) / len(all_response_times),
            sum(all_cascade_rates) / len(all_cascade_rates)
        )

    @staticmethod
    def run_scheduler(env, scheduler):
        """Runs a baseline scheduler over all tasks. Returns metrics."""
        for index, task in enumerate(env.tasks):
            env.current_time = task.arrival_time
            feasible_vms = env.get_feasible_vms(task)
            if not feasible_vms:
                continue
            vm = scheduler.select_vm(feasible_vms, task)
            env.assign_task(task, vm)
            
            # if index < 5:
            #     print(f"Task {index}: exe={task.execution_time:.2f}, wait={task.waiting_time:.2f}, response={task.response_time:.2f}, deadline={task.deadline:.2f}, success={task.success}")
            # if index < 5:
            #     print(f"Task {index}: selected vm={vm.vm_id}, vm.available_time={vm.available_time:.4f}, task.arrival_time={task.arrival_time:.4f}, wait={task.waiting_time:.4f}")
        return Utils.calculate_metrics(env.tasks, 0)

    @staticmethod
    def calculate_metrics(tasks, total_cascade_misses):
        """Computes success rate, average response time, and cascade miss rate."""
        total = len(tasks)
        success_rate = sum(t.success for t in tasks) / total
        avg_response_time = sum(t.response_time for t in tasks) / total
        cascade_rate = total_cascade_misses / total
        return success_rate, avg_response_time, cascade_rate

