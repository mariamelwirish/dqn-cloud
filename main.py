import matplotlib.pyplot as plt
from environment import CloudEnvironment
from algorithms.dqn.agent import DQNAgent
import numpy as np
from algorithms.random.scheduler import RandomScheduler
from utils import Utils


arrival_rates = [10, 15, 20, 25, 30]

results = {
    'dqn': {'success_rate': [], 'avg_cost': [], 'avg_response_time': []},
    'random': {'success_rate': [], 'avg_cost': [], 'avg_response_time': []},
}

for rate in arrival_rates:
    print(f"\nRunning simulation for arrival rate = {rate}")

    # create environment and agent for each rate
    seed = np.random.default_rng(1)
    env = Utils.create_environment(seed, rate)
    success_rate, avg_cost, avg_response_time = Utils.run_dqn(env)
    results['dqn']['success_rate'].append(success_rate)
    results['dqn']['avg_cost'].append(avg_cost)
    results['dqn']['avg_response_time'].append(avg_response_time)
    print("DQN Scheduler:")
    print(f"  Success Rate: {success_rate*100:.2f}%")
    print(f"  Average Cost: {avg_cost:.4f}")
    print(f"  Average Response Time: {avg_response_time:.4f}")

    # Random 
    seed = np.random.default_rng(1)
    env = Utils.create_environment(seed, rate)
    success_rate, avg_cost, avg_response_time = Utils.run_scheduler(env, RandomScheduler(seed))
    results['random']['success_rate'].append(success_rate)
    results['random']['avg_cost'].append(avg_cost)
    results['random']['avg_response_time'].append(avg_response_time)
    print("Random Scheduler:")
    print(f"  Success Rate: {success_rate*100:.2f}%")
    print(f"  Average Cost: {avg_cost:.4f}")
    print(f"  Average Response Time: {avg_response_time:.4f}")


# plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

x = np.arange(len(arrival_rates))
width = 0.2
n = len(results)

metrics_to_plot = [
    ('success_rate', 'Success Rate', 'Success Rate'),
    ('avg_cost', 'Average Cost', 'Cost'),
    ('avg_response_time', 'Average Response Time', 'Response Time'),
]

for ax, (metric, title, ylabel) in zip(axes, metrics_to_plot):
    for i, (algorithm, metrics) in enumerate(results.items()):
        offset = (i - n/2) * width + width/2
        ax.bar(x + offset, metrics[metric], width, label=algorithm, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Mean Arrival Rate')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(arrival_rates)
    ax.legend()

plt.tight_layout()
plt.savefig('results_dqn.png')
plt.show()

print("\nDone! Results saved to results_dqn.png")