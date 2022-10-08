import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
import numpy as np

class PPOMemory(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

    def push(self, state, action, reward, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

    def get_tensors(self, device):
        return torch.FloatTensor(np.array(self.states)).to(device), \
            torch.FloatTensor(np.array(self.actions)).to(device), \
            torch.FloatTensor(np.array(self.rewards)).to(device), \
            torch.FloatTensor(self.log_probs).to(device)

class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        return self.actor(x)

class Critic(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden_dim=64):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        return self.critic(x)

class PPO(object):
    def __init__(self, env, cfg):
        self.actor_lr = cfg.get('actor_lr', 0.1)
        self.critic_lr = cfg.get('critic_lr', 0.1)
        self.gamma = cfg.get('gamma', 0.99)
        self.clip_eps = cfg.get('clip_eps', 0.95)
        self.coef_entropy = cfg.get('coef_entropy', 0.01)
        self.run_epochs = cfg.get('run_epochs', 5)
        self.device = cfg.get('device', 'cpu')
        
        self.memory = PPOMemory()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def choose_action(self, state):
        """采样一个动作
        """
        state = torch.FloatTensor(state).to(self.device)
        p_actions = self.actor(state)
        # p_actions = F.softmax(p_actions, dim=-1)
        dist = Categorical(logits=p_actions)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def evaluate(self, states, actions):
        """计算已经采样数据的values, log_probs, entropy
        """
        p_actions = self.actor(states)
        # p_actions = F.softmax(p_actions, dim=-1)
        dist = Categorical(logits=p_actions)

        log_probs = dist.log_prob(actions)
        values = self.critic(states)
        dist_entropy = dist.entropy()
        return log_probs, values, dist_entropy

    def update(self):
        """训练actor和critic网络，根据已经采样好的数据跑run_epochs个epoch
        """
        # 获取memory中的数据
        old_states, old_actions, old_rewards, old_log_probs = self.memory.get_tensors(self.device)

        # Monte-Carlo计算discount后的rewards，这里没有使用GAE
        running_add = 0  # 这句实际上默认了最后一个数据刚好是done的
        discounted_rewards = torch.zeros_like(old_rewards)
        for t in reversed(range(len(self.memory.rewards))):
            if self.memory.dones[t]:
                running_add = 0
            running_add = running_add * self.gamma + self.memory.rewards[t]
            discounted_rewards[t] = running_add

        _, old_values, _ = self.evaluate(old_states, old_actions)
        old_values = old_values.squeeze().detach()
        advantages = discounted_rewards - old_values

        # 标准化，这步原论文中没有，理论上应该也不是必须
        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-10)
        # discounted_rewards = torch.mean(discounted_rewards) / (torch.std(discounted_rewards) + 1e-10)

        # 用已经采样的数据训练多个epoch
        for _ in range(self.run_epochs):
            # 使用当前的actor-critic网络进行计算
            log_probs, values, entropy = self.evaluate(old_states, old_actions)
            values = values.squeeze()

            # 计算权重p_theta / p_theta_old
            ratios = torch.exp(log_probs - old_log_probs)

            # 计算actor loss
            # advantages = discounted_rewards - values
            actor_loss1 = ratios * advantages
            actor_loss2 = torch.clamp(ratios, min=1-self.clip_eps, max=1+self.clip_eps) * advantages
            actor_loss = torch.min(actor_loss1, actor_loss2).mean()

            # 计算critic loss
            critic_loss = F.mse_loss(values, discounted_rewards)

            # 计算总loss
            loss = -actor_loss + 0.5 * critic_loss - self.coef_entropy * entropy.mean()

            # 反向传播和梯度下降
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        # 清空使用完的数据
        self.memory.clear()
