import numpy as np
import matplotlib.pyplot as plt
from utils import Utils
from environment import Task, VM
from algorithms.random.scheduler import RandomScheduler
from algorithms.round_robin.scheduler import RoundRobinScheduler
from algorithms.earliest.scheduler import EarliestVMScheduler


# --- Experiment Parameters ---
ARRIVAL_RATES = [10, 15, 20, 25, 30]

SEED_NUMBER = 1
N_EPOCHS = 10

# --- Results Storage ---
results = {
    'DQN':        {'success_rate': [], 'avg_response_time': [], 'cascade_rate': []},
    'Random':     {'success_rate': [], 'avg_response_time': [], 'cascade_rate': []},
    'Round Robin':{'success_rate': [], 'avg_response_time': [], 'cascade_rate': []},
    'Earliest':   {'success_rate': [], 'avg_response_time': [], 'cascade_rate': []},
}

# --- Run Experiments ---
for rate in ARRIVAL_RATES:
    print(f"\nRunning experiments for arrival rate = {rate}")

    # --- DQN ---
    seed = np.random.default_rng(SEED_NUMBER)
    Task.reset_counter()
    VM.reset_counter()
    env = Utils.create_environment(seed, rate)
    success_rate, avg_response_time, cascade_rate = Utils.run_dqn(env, seed, N_EPOCHS)
    results['DQN']['success_rate'].append(success_rate)
    results['DQN']['avg_response_time'].append(avg_response_time)
    results['DQN']['cascade_rate'].append(cascade_rate)
    print(f"  DQN          → Success: {success_rate:.2%} | Response: {avg_response_time:.4f} | Cascade: {cascade_rate:.4f}")

    # --- Random ---
    seed = np.random.default_rng(SEED_NUMBER)
    Task.reset_counter()
    VM.reset_counter()
    env = Utils.create_environment(seed, rate)
    success_rate, avg_response_time, cascade_rate = Utils.run_scheduler(env, RandomScheduler(seed))
    results['Random']['success_rate'].append(success_rate)
    results['Random']['avg_response_time'].append(avg_response_time)
    results['Random']['cascade_rate'].append(cascade_rate)
    print(f"  Random       → Success: {success_rate:.2%} | Response: {avg_response_time:.4f} | Cascade: {cascade_rate:.4f}")

    # --- Round Robin ---
    seed = np.random.default_rng(SEED_NUMBER)
    Task.reset_counter()
    VM.reset_counter()
    env = Utils.create_environment(seed, rate)
    success_rate, avg_response_time, cascade_rate = Utils.run_scheduler(env, RoundRobinScheduler())
    results['Round Robin']['success_rate'].append(success_rate)
    results['Round Robin']['avg_response_time'].append(avg_response_time)
    results['Round Robin']['cascade_rate'].append(cascade_rate)
    print(f"  Round Robin  → Success: {success_rate:.2%} | Response: {avg_response_time:.4f} | Cascade: {cascade_rate:.4f}")

    # --- Earliest ---
    seed = np.random.default_rng(SEED_NUMBER)
    Task.reset_counter()
    VM.reset_counter()
    env = Utils.create_environment(seed, rate)
    success_rate, avg_response_time, cascade_rate = Utils.run_scheduler(env, EarliestVMScheduler())
    results['Earliest']['success_rate'].append(success_rate)
    results['Earliest']['avg_response_time'].append(avg_response_time)
    results['Earliest']['cascade_rate'].append(cascade_rate)
    print(f"  Earliest     → Success: {success_rate:.2%} | Response: {avg_response_time:.4f} | Cascade: {cascade_rate:.4f}")

print("\nAll experiments done. Plotting results...")

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

x = np.arange(len(ARRIVAL_RATES))
width = 0.2
colors = ['#2196F3', '#FF5722', '#4CAF50', '#FF9800']

metrics = [
    ('success_rate',    'Success Rate',          'Success Rate (%)'),
    ('avg_response_time','Average Response Time', 'Response Time (s)'),
    # ('cascade_rate',    'Cascade Miss Rate',      'Cascade Miss Rate'),
]

for ax, (metric, title, ylabel) in zip(axes, metrics):
    for i, (scheduler, color) in enumerate(zip(results.keys(), colors)):
        offset = (i - len(results) / 2) * width + width / 2
        values = results[scheduler][metric]
        if metric == 'success_rate':
            values = [v * 100 for v in values]
        ax.bar(x + offset, values, width, label=scheduler, color=color, alpha=0.85)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean Arrival Rate', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(ARRIVAL_RATES)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('DQN-Cloud Scheduler Performance Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('results.png', dpi=150)
plt.show()

print("Results saved to results.png")