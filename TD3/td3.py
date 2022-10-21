import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import random
import copy

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

class GaussianNoise(object):
    def __init__(self, mu, sigma, dim):
        self.mu = mu
        self.sigma = sigma
        self.dim = dim

    def sample(self):
        return self.mu + self.sigma * np.random.randn(self.dim)

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x):
        out = self.net(x)
        return torch.tanh(out)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net1(x), self.net2(x)

    def Q1(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net1(x)

class TD3(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = cfg.get('batch_size', 256)
        self.device = cfg.get('device', 'cpu')
        
        self.actor = Actor(self.state_dim, cfg.get('hidden_dim', 256), self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim, cfg.get('hidden_dim', 256), self.action_dim).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.get('actor_lr', 3e-4))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.get('critic_lr', 3e-4))
        
        self.buffer = ReplayBuffer(cfg.get('buffer_capacity', 1000000))
        self.noise = GaussianNoise(0, cfg.get('noise_sigma', 0.1), self.action_dim)
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.005)
        self.policy_noise = cfg.get('policy_noise', 0.2)
        self.noise_clip = cfg.get('noise_clip', 0.5)
        self.policy_freq = cfg.get('policy_freq', 2)

        self.total_iters = 0

    def sample_action(self, state):
        """train时使用, 有噪声
        """
        state = torch.FloatTensor(state).to(self.device)
        action = np.clip(self.actor(state).cpu().data.numpy() + self.noise.sample(), -1.0, 1.0)
        return action
    
    def predict_action(self, state):
        """test时使用, 没有噪声
        """
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy()
        return action

    def soft_update(self, net, target_net):
        for target_param, param, in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.LongTensor(dones).to(self.device)

        # 计算critic_loss
        with torch.no_grad():
            noise = torch.clamp(
                self.policy_noise * torch.randn_like(actions),
                -self.noise_clip, self.noise_clip
            )
            next_actions = torch.clamp(
                self.target_actor(next_states) + noise,
                -1.0, 1.0
            )
            target_Q1, target_Q2 = self.target_critic(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2).squeeze()
            target_Q = rewards + self.gamma * target_Q * (1 - dones)
        
        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1.squeeze(), target_Q) + F.mse_loss(current_Q2.squeeze(), target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_iters % self.policy_freq == 0:
            # 计算actor_loss
            actor_loss = -torch.mean(self.critic.Q1(states, self.actor(states)))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 更新目标网络
            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic, self.target_critic)

    def save(self, cfg, steps):
        import os.path
        dir_path = "{}/{}_{}_{}_{}/".format(
            cfg['model_path'], cfg['algo_name'], cfg['env_name'], cfg['seed'], steps)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.actor.state_dict(), os.path.join(dir_path, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(dir_path, "critic.pt"))

    def load(self, cfg, steps):
        import os.path
        dir_path = "{}/{}_{}_{}_{}/".format(
            cfg['model_path'], cfg['algo_name'], cfg['env_name'], cfg['seed'], steps)
        self.actor.load_state_dict(torch.load(os.path.join(dir_path, "actor.pt")))
        self.target_actor = copy.deepcopy(self.actor)
        self.critic.load_state_dict(torch.load(os.path.join(dir_path, "critic.pt")))
        self.target_critic = copy.deepcopy(self.critic)
        