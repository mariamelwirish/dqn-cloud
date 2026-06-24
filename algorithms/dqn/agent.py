import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from algorithms.dqn.network import DQNNetwork

class DQNAgent:
    def __init__(self,
                input_size, hidden_size, output_size, seed,
                learning_rate=0.01, gamma=0.9,
                epsilon=0.9, epsilon_decay=0.0002, epsilon_min=0.01,
                batch_size=30, memory_size=800, target_update_freq=50):
        self.online_network = DQNNetwork(input_size, hidden_size, output_size)
        self.target_network = DQNNetwork(input_size, hidden_size, output_size)
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.memory = deque(maxlen=memory_size)

        self.steps = 0

        self.seed = seed
        torch.manual_seed(int(seed.integers(0, 2**32)))
        self.optimizer = optim.RMSprop(self.online_network.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_min = epsilon_min

    def select_action(self, state, feasible_indices):
        # Explore
        if self.seed.random() < self.epsilon:
            return int(self.seed.choice(feasible_indices))

        # Exploit
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)

        # Only consider feasible VM indices
        feasible_q = {idx: q_values[idx].item() for idx in feasible_indices}
        return max(feasible_q, key=feasible_q.get)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self):
        # Step 1: not enough experiences yet
        if len(self.memory) < self.batch_size:
            return

        # Step 2: sample random batch
        indices = self.seed.choice(len(self.memory), size=self.batch_size, replace=False)
        batch = [list(self.memory)[i] for i in indices]
        states, actions, rewards, next_states = zip(*batch)

        # Step 3: convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)

        # Step 4: online network predicts current Q-values
        current_q = self.online_network(states_t).gather(1, actions_t)

        # Step 5: target network predicts best Q-values for next states
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(1)[0].unsqueeze(1)

        # Step 6: compute targets
        target_q = rewards_t + self.gamma * next_q

        # Step 7: compute loss
        loss = nn.MSELoss()(current_q, target_q)

        # Step 8: update online network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        # Step 9: update target network
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

        # Step 10: increase epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
