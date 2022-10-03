import gym
from gym.wrappers.record_video import RecordVideo
import matplotlib.pyplot as plt
from reinforce import Reinforce

class Solver(object):
    @staticmethod
    def create_env_agent(cfg, seed=1):
        env = gym.make(cfg['env_name'])
        env.reset(seed=seed)
        agent = Reinforce(env, cfg)
        return env, agent
    
    @staticmethod
    def train(env, agent, cfg):
        rewards = []
        for ep in range(cfg['train_episodes']):
            state = env.reset()[0]
            ep_reward = 0
            for _ in range(cfg['ep_max_steps']):
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.store_transition(state, action, reward)
                state = next_state
                ep_reward += reward
                if terminated or truncated:
                    break
            agent.learn()  # 每个回合进行一次学习
            rewards.append(ep_reward)
            print(f"Train episode {ep+1}/{cfg['train_episodes']}: reward: {ep_reward}")
        
        return {'rewards': rewards}

    @staticmethod
    def test(env, agent, cfg):
        rewards = []
        for ep in range(cfg['test_episodes']):
            state = env.reset()[0]
            ep_reward = 0
            for _ in range(cfg['ep_max_steps']):
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                state = next_state
                ep_reward += reward
                if terminated or truncated:
                    break
            rewards.append(ep_reward)
            print(f"Test episode {ep+1}/{cfg['test_episodes']}: reward: {ep_reward}")

        return {'rewards': rewards}

    @staticmethod
    def create_rec_env(cfg, tag="", seed=1):
        env = gym.make(cfg['env_name'], render_mode='rgb_array')
        env.reset(seed=seed)
        env = RecordVideo(
            env, './output/results/',
            episode_trigger=lambda a: a == 0,
            name_prefix='cartpole-'+tag
        )  # 只录制第一次test
        return env

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