import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class PPOMemory(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.dones = []

    def push(self, state, action, reward, next_state, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.dones = []

    def get_tensors(self, device):
        return torch.FloatTensor(np.array(self.states)).to(device), \
            torch.FloatTensor(np.array(self.actions)).to(device), \
            torch.FloatTensor(np.array(self.rewards)).to(device), \
            torch.FloatTensor(np.array(self.next_states)).to(device), \
            torch.FloatTensor(self.log_probs).to(device), \
            torch.LongTensor(self.dones).to(device),
    
    def sample_batch(self, batch_size):
        batch_step = np.arange(0, len(self.states), batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_step]
        return batches

class GaussianActor(nn.Module):
    """生成正态分布的actor，返回mu和sigma
    """
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, out_dim)
        self.sigma = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        out = self.actor(x)
        mu = torch.tanh(self.mu(out))
        sigma = F.softplus(self.sigma(out))
        return mu, sigma
    
    def get_dist(self, state):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)
        return dist
    
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
        self.actor_lr = cfg.get('actor_lr', 0.0003)
        self.critic_lr = cfg.get('critic_lr', 0.0003)
        self.gamma = cfg.get('gamma', 0.99)
        self.gae_lambda = cfg.get('gae_lambda', 0.95)
        self.clip_eps = cfg.get('clip_eps', 0.2)
        self.coef_entropy = cfg.get('coef_entropy', 0.01)
        self.entropy_decay = cfg.get('entropy_decay', 0.99)
        self.run_epochs = cfg.get('run_epochs', 5)
        self.batch_size = cfg.get('batch_size', 5)
        self.device = cfg.get('device', 'cpu')
        
        self.action_low = torch.FloatTensor(env.action_space.low).to(self.device)
        self.action_high = torch.FloatTensor(env.action_space.high).to(self.device)
        self.memory = PPOMemory()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = GaussianActor(self.state_dim, self.action_dim, cfg.get('hidden_dim', 128)).to(self.device)
        self.critic = Critic(self.state_dim, 1, cfg.get('hidden_dim', 128)).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    @staticmethod
    def reward_adapter(reward):
        """更改一下reward，可以加速收敛
        """
        return (reward + 8.0) / 8.0

    def choose_action(self, state):
        """采样一个动作
        """
        state = torch.FloatTensor(state).to(self.device)
        dist = self.actor.get_dist(state)
        action = dist.sample()
        action = action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob

    def evaluate(self, states, actions):
        """计算已经采样数据的values, log_probs, entropy
        """
        dist = self.actor.get_dist(states)
        log_probs = dist.log_prob(actions).squeeze()
        values = self.critic(states).squeeze()
        dist_entropy = dist.entropy()
        return log_probs, values, dist_entropy

    def compute_advantages(self, td_deltas):
        td_deltas = td_deltas.cpu().detach().numpy()
        adv_list = []
        advantage = 0.0
        for delta in td_deltas[::-1]:
            advantage = self.gamma * self.gae_lambda * advantage + delta
            adv_list.append(advantage)
        adv_list.reverse()
        return torch.FloatTensor(np.array(adv_list)).to(self.device)

    def update(self):
        """训练actor和critic网络，根据已经采样好的数据跑run_epochs个epoch
        """
        self.coef_entropy *= self.entropy_decay
        # 获取memory中的数据
        states, actions, rewards, next_states, old_log_probs, dones = \
            self.memory.get_tensors(self.device)
        # 更改rewards加速收敛
        rewards = PPO.reward_adapter(rewards)
        
        # 计算advantage(GAE方法)
        with torch.no_grad():
            td_targets = rewards + self.gamma * self.critic(next_states).squeeze() * (1 - dones)
            td_deltas = td_targets - self.critic(states).squeeze()
            advantages = self.compute_advantages(td_deltas)
            # 原论文中没有提标准化，算是一个trick
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 用已经采样的数据训练多个epoch
        for _ in range(self.run_epochs):
            batches = self.memory.sample_batch(self.batch_size)
            for batch in batches:
                # 使用当前的actor-critic网络进行计算采样的数据
                log_probs, critic_values, entropy = self.evaluate(states[batch], actions[batch])

                # 计算权重p_theta / p_theta_old
                ratios = torch.exp(log_probs - old_log_probs[batch])

                # 计算actor loss
                actor_loss1 = ratios * advantages[batch]
                actor_loss2 = torch.clamp(ratios, min=1-self.clip_eps, max=1+self.clip_eps) * advantages[batch]
                actor_loss = -torch.min(actor_loss1, actor_loss2).mean() - self.coef_entropy * entropy.mean()

                # 计算critic loss
                critic_loss = F.mse_loss(critic_values, td_targets.detach()[batch])

                # 反向传播和梯度下降
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) 
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) 
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # 清空使用完的数据
        self.memory.clear()
    
    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'ppo-actor.pt')
        torch.save(self.critic.state_dict(), path + 'ppo-critic.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'ppo-actor.pt'))
        self.critic.load_state_dict(torch.load(path + 'ppo-critic.pt'))