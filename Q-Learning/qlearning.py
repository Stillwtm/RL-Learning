from collections import defaultdict
import math
import numpy as np

class QLearning(object):
    def __init__(self, n_actions, cfg):
        self.n_actions = n_actions
        self.lr = cfg.get('lr', 0.9)
        self.gamma = cfg.get('gamma', 0.9)
        self.epsilon_beg = cfg.get('epsilon_beg', 0.95)
        self.epsilon_end = cfg.get('epsilon_end', 0.01)
        self.epsilon_decay = cfg.get('epsilon_decay', 300)
        self.epsilon = self.epsilon_beg
        self.sample_count = 0
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

    def sample(self, state):
        """train时采样action
        """
        self.sample_count += 1
        # 对epsilon进行指数衰减
        self.epsilon = self.epsilon_beg + (self.epsilon_end - self.epsilon_beg) * \
            math.exp(self.sample_count / self.epsilon_decay)
        
        # epsilon-greedy策略
        if np.random.uniform(0, 1) > self.epsilon:
            # 选择当前Q表里最优的
            action = np.argmax(self.q_table[str(state)])
        else:
            # 随机探索
            action = np.random.choice(self.n_actions)
        return action

    def predict(self, state):
        """test时预测action
        """
        # 直接选择Q表里最优的
        action = np.argmax(self.q_table[str(state)])
        return action

    def update(self, state, action, reward, next_state, done):
        """更新Q表
        """
        q_predict = self.q_table[str(state)][action]
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.q_table[str(next_state)])
        self.q_table[str(state)][action] += self.lr * (q_target - q_predict)

    def save(self, path):
        import dill
        data = open(path + 'Qleaning_model.pkl', 'wb')
        dill.dump(self.q_table, data)
        print('模型保存至 ' + path + 'Qleaning_model.pkl')

    def load(self, path):
        import dill
        data = open(path + 'Qleaning_model.pkl', 'rb')
        self.q_table = dill.load(data)
        print('模型已加载')