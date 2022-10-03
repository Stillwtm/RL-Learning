import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PGNet(nn.Module):
    """Policy网络，这里采用两层的FC
    """
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        out = self.net(x)
        return out

class Reinforce(object):
    def __init__(self, env, cfg):
        self.lr = cfg.get('lr', 0.1)
        self.gamma = cfg.get('gamma', 0.9)
        self.device = cfg.get('device', 'cpu')
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []
        self.policy_net = PGNet(self.n_states, self.n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def choose_action(self, state):
        """将Policy网络的输出作为概率选择一个动作
        """
        state = torch.FloatTensor(state).to(self.device)
        p_actions = self.policy_net(state)
        pd = Categorical(logits=p_actions)
        action = pd.sample().item()
        return action

    def store_transition(self, state, action, reward):
        """储存一个episode内的数据
        """
        self.ep_states.append(state)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)

    def clear_transition(self):
        """清空已经储存的数据
        """
        self.ep_states.clear()
        self.ep_actions.clear()
        self.ep_rewards.clear()

    def learn(self):
        """根据一个episode的数据进行梯度上升
        """
        # 计算G_t
        running_add = 0
        discounted_rewards = torch.zeros(len(self.ep_rewards), dtype=torch.float32, device=self.device)
        for t in reversed(range(len(self.ep_rewards))):
            running_add = self.gamma * running_add + self.ep_rewards[t]
            discounted_rewards[t] = running_add
        
        # 标准化
        # with torch.no_grad():
        reward_mean = torch.mean(discounted_rewards)
        reward_std = torch.std(discounted_rewards, unbiased=False)
        discounted_rewards = (discounted_rewards - reward_mean) / reward_std

        # 写法一，直接利用分类分布进行计算
        self.optimizer.zero_grad()
        for i in range(len(self.ep_rewards)):
            state = torch.FloatTensor(self.ep_states[i]).to(self.device)
            action = torch.FloatTensor([self.ep_actions[i]]).to(self.device)
            reward = discounted_rewards[i]
            probs = self.policy_net(state)
            pd = Categorical(logits=probs)
            loss = -pd.log_prob(action) * reward
            loss.backward()
        self.optimizer.step()
    
        # 写法二，利用交叉熵进行计算，刚好符合-log的需求，已经实验两种方式计算的loss是一样的
        # p_actions = self.policy_net(torch.FloatTensor(self.ep_states).to(self.device))
        # neg_log_p = F.cross_entropy(p_actions, torch.LongTensor(self.ep_actions).to(self.device), reduction='none')
        # loss = torch.mean(neg_log_p * discounted_rewards)
        # # 反向传播，由于上面的loss有负号，故可以利用现成的梯度下降优化器来完成梯度上升
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # 清空这个episode内的数据
        self.clear_transition()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'rf_model.pt')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'rf_model.pt'))