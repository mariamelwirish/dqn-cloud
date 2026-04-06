import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from environment import Job
from algorithms.dqn.network import DQNNetwork

class DQNAgent:
    def __init__(self, input_size, hidden_size, output_size, seed, learning_rate=0.01, discount_factor=0.9, replay_memory_size=800, mini_batch_size=30, target_update_frequency=500, epsilon=0.9, epsilon_decay=0.002, ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.replay_memory_size = replay_memory_size
        self.mini_batch_size = mini_batch_size
        self.target_update_frequency = target_update_frequency
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.steps = 0
        self.seed = seed
        torch.manual_seed(int(seed.integers(0, 2**32)))

        self.online_network = DQNNetwork(input_size, hidden_size, output_size)
        self.target_network = DQNNetwork(input_size, hidden_size, output_size)
        self.target_network.load_state_dict(self.online_network.state_dict())
        
        self.memory = deque(maxlen=replay_memory_size)

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    # select an action based on epsilon-greedy algorithm (exploration vs exploitation)
    def select_action(self, state):
        # exploration
        if self.seed.random() < self.epsilon:
            return self.seed.integers(0, self.output_size)
        # exploitation
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                q_values = self.online_network(state_tensor)
                return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < self.mini_batch_size:
            return
        
        # Sample a mini-batch from the replay memory
        indices = self.seed.choice(len(self.memory), size=self.mini_batch_size, replace=False)
        mini_batch = [list(self.memory)[i] for i in indices]
        states, actions, rewards, next_states = zip(*mini_batch)

        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)

        # Compute current Q values
        current_q_values = self.online_network(states_tensor).gather(1, actions_tensor)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0].unsqueeze(1)
        
        # Q-learning update rule: Q(s, a) = r + gamma * max(Q(s', a'))
        target_q_values = rewards_tensor + self.discount_factor * next_q_values

        # Compute loss and update the online network
        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1

        # Update target network
        if self.steps % self.target_update_frequency == 0:
            self.update_target_network()
        # Decay epsilon
        if self.epsilon > 0.01:
            self.epsilon -= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

