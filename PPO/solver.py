import gym
from gym.wrappers.record_video import RecordVideo
import matplotlib.pyplot as plt
import torch
import numpy as np
from ppo import PPO

class Solver(object):
    @staticmethod
    def create_env_agent(cfg, seed=1, unwrap=False):
        env = gym.make(cfg['env_name'])
        if unwrap:
            env = env.unwrapped
        agent = PPO(env, cfg)
        # 设置随机种子
        env.reset(seed=seed)
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)
        return env, agent
    
    @staticmethod
    def train(env, agent, cfg, last_steps=0):
        rewards = []
        steps=last_steps
        for ep in range(cfg['train_episodes']):
            state = env.reset()[0]
            ep_reward = 0
            for _ in range(cfg['ep_max_steps']):
                action, log_prob = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.memory.push(state, action, reward, log_prob, terminated or truncated)
                state = next_state
                ep_reward += reward
                steps += 1
                if terminated or truncated:
                    break
            if steps >= cfg['steps_per_batch']:
                agent.update()
                steps = 0
            rewards.append(ep_reward)
            print(f"Train episode {ep+1}/{cfg['train_episodes']}: reward: {ep_reward}")
        
        return {'rewards': rewards, 'last_steps': steps}

    @staticmethod
    def test(env, agent, cfg):
        rewards = []
        for ep in range(cfg['test_episodes']):
            state = env.reset()[0]
            ep_reward = 0
            for _ in range(cfg['ep_max_steps']):
                action, _ = agent.choose_action(state)
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
            name_prefix=cfg['env_name']+'-'+tag
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