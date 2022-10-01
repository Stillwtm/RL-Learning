import matplotlib.pyplot as plt
from qlearning import QLearning
import gym

class Solver(object):
    @staticmethod
    def env_agent_config(cfg, seed=0):
        """创建env和agent对象
        """
        env = gym.make(cfg['env_name'])
        env.reset(seed=seed)  # 设置随机种子
        n_states = env.observation_space.n  # 状态维度
        n_actions = env.action_space.n  # 动作维度
        agent = QLearning(n_actions, cfg)
        return env, agent

    @staticmethod
    def train(env, agent, cfg):
        rewards = []
        for i in range(cfg['train_episodes']):
            state = env.reset()[0]
            done = False
            ep_reward = 0
            while not done:
                action = agent.sample(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                ep_reward += reward
                state = next_state
                done = terminated or truncated
                
            rewards.append(ep_reward)
            print(f"Train episode {i+1}/{cfg['train_episodes']}: reward: {ep_reward}")
        
        return {'rewards': rewards}

    @staticmethod
    def test(env, agent, cfg, max_iters=500):
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

    @staticmethod
    def plot(rewards, cfg, tag='test', save=False, path=None):
        fig = plt.figure()
        plt.title("learning curve on {} of {} for {}".format(
            cfg['device'], cfg['algo_name'], cfg['env_name']))
        plt.xlabel('Epsiodes')
        plt.plot(rewards, label='rewards')
        plt.legend()
        if save:
            plt.savefig(path + tag + str(cfg['train_episodes']))
        else:
            plt.show()


