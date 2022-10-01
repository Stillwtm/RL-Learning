import gym
from gym.wrappers.record_video import RecordVideo
from solver import Solver

cfg = {
    'algo_name': 'Q-Learning',
    'env_name': 'CliffWalking-v0',
    'train_episodes': 300,
    'test_episodes': 20,
    'lr': 0.1,
    'gamma': 0.9,
    'epsilon_beg': 0.95,
    'epsilon_end': 0.01,
    'epsilon_decay': 300,
    'device': 'cpu',
}


for train_eps in [50, 100, 150, 200, 250, 300]:
    cfg['train_episodes'] = train_eps

    env, agent = Solver.env_agent_config(cfg)
    res_dict = Solver.train(env, agent, cfg)
    Solver.plot(res_dict['rewards'], cfg, tag='train', save=True, path='./output/results')

    env = gym.make(cfg['env_name'], render_mode='rgb_array')  # 录制视频
    env = RecordVideo(
        env, './output/',
        episode_trigger=lambda a: a == 0,
        name_prefix='cliff-walking-'+str(train_eps)
    )  # 只录制第一次test
    
    res_dict = Solver.test(env, agent, cfg)
    Solver.plot(res_dict['rewards'], cfg, tag='test', save=True, path='./output/results')

env.close()