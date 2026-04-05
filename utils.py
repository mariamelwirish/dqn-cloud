from environment import CloudEnvironment
from algorithms.dqn.agent import DQNAgent

class Utils:
    @staticmethod
    def create_environment(seed, arrival_rate):
        return CloudEnvironment(
            n_vms=10, 
            n_jobs=8000, 
            io_vm_ratio=0.5, 
            io_job_ratio=0.5, 
            mean_job_arrival_rate=arrival_rate, 
            mean_job_length=200, 
            variance_job_length=40,
            seed=seed
        )

    @staticmethod
    def run_dqn(env):
        agent = DQNAgent(input_size=12, hidden_size=20, output_size=10, seed=env.seed)

        for index, job in enumerate(env.jobs):
            env.current_time = job.arrival_time

            state = env.get_state(job)
            action = agent.select_action(state)

            selected_vm = env.vms[action]
            selected_vm.assign_job(job)
            reward = env.calculate_reward(job)

            if index < len(env.jobs) - 1:
                next_state = env.get_state(env.jobs[index + 1])
            else:
                next_state = [0] * 12

            agent.remember(state, action, reward, next_state)
            agent.train()

            if index % 1000 == 0:
                print(f"  Job {index}/{len(env.jobs)} | Epsilon: {agent.epsilon:.3f}")
            
        success_rate, avg_cost, avg_response_time = Utils.calculate_metrics(env.jobs)
        return success_rate, avg_cost, avg_response_time

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