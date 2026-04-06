from environment import CloudEnvironment
from algorithms.dqn.agent import DQNAgent

class Utils:
    @staticmethod
    def create_environment(seed, arrival_rate):
        return CloudEnvironment(
            n_vms=10, 
            n_jobs=8000, 
            mean_job_arrival_rate=arrival_rate, 
            mean_job_length=200,
            job_len_std=20,
            io_job_ratio=0.5,
            seed=seed
        )

    @staticmethod
    def run_dqn(env, n_epochs=5):
        agent = DQNAgent(input_size=11, hidden_size=20, output_size=10, seed=env.seed)
        
        all_success_rates = []
        all_avg_costs = []
        all_avg_response_times = []

        for epoch in range(n_epochs):
            print(f"  Epoch {epoch+1}/{n_epochs}")
            
            # regenerate workload each epoch
            env.jobs = env.generate_jobs()
            # reset VMs
            for vm in env.vms:
                vm.queue = []
                vm.available_time = 0

            for index, job in enumerate(env.jobs):
                env.current_time = job.arrival_time

                state = env.get_state(job)
                action = agent.select_action(state)

                selected_vm = env.vms[action]
                selected_vm.assign_job(job)
                reward = env.calculate_reward(job, selected_vm)

                if index < len(env.jobs) - 1:
                    next_state = env.get_state(env.jobs[index + 1])
                else:
                    next_state = [0] * 11

                agent.remember(state, action, reward, next_state)
                agent.train()

            # skip epoch 0
            if epoch > 0:
                success_rate, avg_cost, avg_response_time = Utils.calculate_metrics(env.jobs)
                all_success_rates.append(success_rate)
                all_avg_costs.append(avg_cost)
                all_avg_response_times.append(avg_response_time)

        return (
            sum(all_success_rates) / len(all_success_rates),
            sum(all_avg_costs) / len(all_avg_costs),
            sum(all_avg_response_times) / len(all_avg_response_times)
        )

    @staticmethod
    def run_scheduler(env, scheduler):
        for index, job in enumerate(env.jobs):
            vm = scheduler.select_vm(env.vms)
            vm.assign_job(job)
        success_rate, avg_cost, avg_response_time = Utils.calculate_metrics(env.jobs)
        return success_rate, avg_cost, avg_response_time

    @staticmethod
    def calculate_metrics(jobs):
        total_jobs = len(jobs)
        success_rate = sum(job.success for job in jobs) / total_jobs
        avg_cost = sum(job.cost for job in jobs) / total_jobs
        avg_response_time = sum(job.response_time for job in jobs) / total_jobs
        return success_rate, avg_cost, avg_response_time