import matplotlib.pyplot as plt
from environment import CloudEnvironment
from algorithms.dqn.agent import DQNAgent

arrival_rates = [10, 15, 20, 25, 30]

results = {
    'success_rate': [],
    'avg_cost': [],
    'avg_response_time': []
}

for rate in arrival_rates:
    print(f"\nRunning simulation for arrival rate = {rate}")
    
    # create fresh environment and agent for each rate
    env = CloudEnvironment(
        n_vms=10, 
        n_jobs=8000, 
        io_vm_ratio=0.5, 
        io_job_ratio=0.5, 
        mean_job_arrival_rate=rate, 
        mean_job_length=200, 
        variance_job_length=40
    )
    agent = DQNAgent(input_size=12, hidden_size=20, output_size=10)

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
            print(f"  Job {index}/8000 | Epsilon: {agent.epsilon:.3f}")

    # calculate metrics
    total_jobs = len(env.jobs)
    success_rate = sum(job.success for job in env.jobs) / total_jobs
    avg_cost = sum(job.cost for job in env.jobs) / total_jobs
    avg_response_time = sum(job.response_time for job in env.jobs) / total_jobs

    results['success_rate'].append(success_rate)
    results['avg_cost'].append(avg_cost)
    results['avg_response_time'].append(avg_response_time)

    print(f"  Success Rate: {success_rate*100:.2f}%")
    print(f"  Average Cost: {avg_cost:.4f}")
    print(f"  Average Response Time: {avg_response_time:.4f}")

# plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(arrival_rates, results['success_rate'], color='blue', alpha=0.7)
axes[0].set_title('Success Rate')
axes[0].set_xlabel('Mean Arrival Rate')
axes[0].set_ylabel('Success Rate')

axes[1].bar(arrival_rates, results['avg_cost'], color='green', alpha=0.7)
axes[1].set_title('Average Cost')
axes[1].set_xlabel('Mean Arrival Rate')
axes[1].set_ylabel('Cost')

axes[2].bar(arrival_rates, results['avg_response_time'], color='red', alpha=0.7)
axes[2].set_title('Average Response Time')
axes[2].set_xlabel('Mean Arrival Rate')
axes[2].set_ylabel('Response Time')

plt.tight_layout()
plt.savefig('results_dqn.png')
plt.show()

print("\nDone! Results saved to results_dqn.png")