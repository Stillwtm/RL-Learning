import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import random
import copy
import math

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """从buffer中采样数据
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class DQN(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = cfg.get('batch_size', 64)
        self.device = cfg.get('device', 'cpu')

        self.buffer = ReplayBuffer(cfg.get('buffer_capacity', 100000))
        self.q_net = QNet(self.state_dim, cfg.get('hidden_dim', 64), self.action_dim).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.get('lr', 1e-3))
        self.gamma = cfg.get('gamma', 0.99)
        self.epsilon = cfg.get('epsilon_start', 0.95)
        self.epsilon_start = cfg.get('epsilon_start', 0.95)
        self.epsilon_end = cfg.get('epsilon_end', 0.01)
        self.epsilon_decay = cfg.get('epsilon_decay', 500)
        self.update_every = cfg.get('update_every', 50)
        self.update_count = 0
        self.sample_count = 0

    def sample_action(self, state):
        """train时使用,e-greedy策略
        """
        self.sample_count += 1
        # epsilon指数衰减
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-self.sample_count / self.epsilon_decay)
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            state = torch.FloatTensor(state).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    
    def predict_action(self, state):
        """test时使用
        """
        state = torch.FloatTensor(state).to(self.device)
        action = self.q_net(state).argmax().item()
        return action

    def update(self):
        if (self.buffer.size() < self.batch_size):
            return
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.LongTensor(dones).to(self.device)

        with torch.no_grad():
            max_next_q_value = self.target_q_net(next_states).max(1)[0]
            q_targets = rewards + self.gamma * max_next_q_value * (1 - dones)
        
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        if self.update_count % self.update_every == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.update_count += 1

    def save(self, cfg, steps):
        torch.save(self.q_net.state_dict(), "{}/{}_{}_{}.pt".format(
            cfg['model_path'], cfg['algo_name'], cfg['env_name'], steps))

    def load(self, cfg, steps):
        self.q_net.load_state_dict(torch.load("{}/{}_{}_{}.pt".format(
            cfg['model_path'], cfg['algo_name'], cfg['env_name'], steps)))
        self.target_q_net.load_state_dict(torch.load("{}/{}_{}_{}.pt".format(
            cfg['model_path'], cfg['algo_name'], cfg['env_name'], steps)))
        