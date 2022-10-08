import matplotlib.pyplot as plt
from sarsa import Sarsa
import gym

class Solver(object):
    @staticmethod
    def env_agent_config(cfg, seed=0):
        """创建env和agent对象
        """
        env = gym.make(cfg['env_name'])
        env.reset(seed=seed)  # 设置随机种子
        n_actions = env.action_space.n  # 动作维度
        agent = Sarsa(n_actions, cfg)
        return env, agent

    @staticmethod
    def train(env, agent, cfg):
        rewards = []
        for i in range(cfg['train_episodes']):
            state = env.reset()[0]
            done = False
            ep_reward = 0
            action = agent.sample(state)
            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_action = agent.sample(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
                ep_reward += reward
                
            rewards.append(ep_reward)
            print(f"Train episode {i+1}/{cfg['train_episodes']}: reward: {ep_reward}")
        
        return {'rewards': rewards}

    @staticmethod
    def test(env, agent, cfg, max_iters=150):
        rewards = []
        for i in range(cfg['test_episodes']):
            state = env.reset()[0]
            done = False
            ep_reward = 0
            iters = 0
            while not done and iters < max_iters:
                action = agent.predict(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                state = next_state
                done = terminated or truncated
                iters += 1
            rewards.append(ep_reward)
            print(f"Test episode {i+1}/{cfg['test_episodes']}: reward: {ep_reward}")
        
        return {'rewards': rewards}

class Drawer(object):
    @staticmethod
    def smooth(data, weight=0.9):  
        '''用于平滑曲线，weight越大越平滑
        '''
        last = data[0]
        smoothed = []
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
            smoothed.append(smoothed_val)                    
            last = smoothed_val                                
        return smoothed

    @staticmethod
    def plot(data, cfg, tag='test', save=False, path=None):
        plt.figure()
        plt.title("learning curve on {} of {} for {}".format(
            cfg['device'], cfg['algo_name'], cfg['env_name']))
        plt.xlabel('Epsiodes')
        plt.plot(data, label='rewards')
        plt.plot(Drawer.smooth(data), label='smoothed_rewards')
        plt.legend()
        if save:
            plt.savefig(path + tag)
        else:
            plt.show()



