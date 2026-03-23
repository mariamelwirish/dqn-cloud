# DQN-Cloud: Cost-Aware Cloud Job Scheduling

A Python/PyTorch implementation of a Deep Reinforcement Learning-based scheduler for cost-aware cloud job scheduling.

📓 [Simulator Documentation & Research Notes](https://mariamelwirish.notion.site/Simulator-Documentation-32b72029a06f80f1b72fd5d0e2f5fb6d?source=copy_link)

---

## What This Project Does

Cloud job scheduling is the problem of deciding which virtual machine (VM) should run each incoming job. Bad scheduling leads to missed deadlines, unhappy users, and unnecessarily high costs.

DQN-Cloud is a DQN-based scheduler that learns to make smarter scheduling decisions by optimizing for both **job success rate** (meeting deadlines) and **execution cost**.

---

## Project Structure

```
dqn-cloud/
│
├── environment.py               # Cloud simulation: Jobs, VMs, reward function
│
├── algorithms/
│   └── dqn/
│       ├── agent.py             # DQN agent (epsilon-greedy, replay buffer, training loop)
│       └── network.py           # Neural network architecture (Q-network)
│
└── main.py                      # Runs experiments and generates plots
```

### Key Files Explained

**`environment.py`**
Defines the simulation. A `CloudEnvironment` holds a set of VMs and a stream of jobs. Each job has a compute requirement (`req_com`), a type (CPU or I/O), and a deadline (`qos`). Each VM has a speed (`v_com`), a type, and a cost rate (`v_cost`). The environment tracks waiting time, execution time, response time, and whether the job met its deadline.

**`algorithms/dqn/agent.py`**
The DQN agent observes the current job and the queue state of all VMs, then picks the best VM to assign the job to. It learns through experience replay and a target network.

**`algorithms/dqn/network.py`**
A feedforward neural network that takes the state (job features + VM waiting times) as input and outputs a Q-value for each possible VM (action).

**`main.py`**
Runs the full experiment suite — sweeps over different mean job arrival rates and collects success rate, average cost, and average response time for each scheduler.

---

## Setup

### Requirements

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib

### Install Dependencies

**Windows:**
```bash
pip install torch numpy matplotlib
```

**Mac/Linux:**
```bash
pip3 install torch numpy matplotlib
```

> If you're using a virtual environment (recommended):
>
> **Windows:** `python -m venv venv` → `venv\Scripts\activate`
>
> **Mac/Linux:** `python3 -m venv venv` → `source venv/bin/activate`

---

## Running Experiments

**Windows:**
```bash
python main.py
```

**Mac/Linux:**
```bash
python3 main.py
```

This will run experiments across varying mean job arrival rates (10, 15, 20, 25, 30) and produce bar charts comparing success rate, average cost, and average response time.

---

## Simulation Parameters

| Parameter | Value | Description |
|---|---|---|
| `n_vms` | 10 | Number of virtual machines |
| `n_jobs` | 8000 | Total jobs per experiment |
| `v_com` | Uniform(1500, 2500) | VM computing speed (instructions/sec) |
| `v_cost` | Uniform(3, 5) | VM cost per time unit |
| `req_com` | Normal(200, √40) | Job size (number of instructions) |
| `qos` | exe_estimate × Uniform(2, 4) | Job deadline |
| `io_ratio` | 0.5 | Half of jobs and VMs are I/O type |

---

## Results

Experiments sweep over **mean job arrival rate** (jobs per second). As arrival rate increases:

- **Success rate decreases** — more jobs arrive faster than the system can handle them, so more miss their deadlines.
- **Average response time increases** — queues build up, so jobs wait longer.
- **Average cost increases** — more jobs are assigned to mismatched VMs (CPU job on I/O VM), doubling execution time and therefore cost.

DQN learns to make better assignment decisions than random or naive baselines by considering both deadline urgency and VM availability.

---

## Baselines (In Progress)

Three simple schedulers are planned for comparison:

- **Random** — picks a VM uniformly at random.
- **Round Robin** — cycles through VMs in order, regardless of load.
- **Earliest Available** — always picks the VM that will be free soonest.

---

## Notes on Reproducibility

The simulation uses `np.random.seed(42)` before VM creation and job generation inside each `CloudEnvironment` to ensure all schedulers are evaluated on the same workload. Without this, different experiments would generate different jobs, making comparisons meaningless.

---