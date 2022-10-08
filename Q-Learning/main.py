import gym
from gym.wrappers.record_video import RecordVideo
from solver import Solver, Drawer

cfg = {
    'algo_name': 'Q-Learning',
    'env_name': 'CliffWalking-v0',
    'train_episodes': 50,
    'test_episodes': 20,
    'lr': 0.1,
    'gamma': 0.9,
    'epsilon_beg': 0.95,
    'epsilon_end': 0.01,
    'epsilon_decay': 300,
    'device': 'cpu',
}


env, agent = Solver.env_agent_config(cfg)
train_rewards = []
for i in range(4):
    # 训练
    res_dict = Solver.train(env, agent, cfg)
    train_rewards += res_dict['rewards']
    # 测试+录像
    env_rec = gym.make(cfg['env_name'], render_mode='rgb_array')  # 录制视频
    env_rec = RecordVideo(
        env_rec, './output/results/',
        episode_trigger=lambda a: a == 0,
        name_prefix='cliff-walking-'+str((i+1) * cfg['train_episodes'])
    )  # 只录制第一次test
    res_dict = Solver.test(env_rec, agent, cfg)
    Drawer.plot(
        res_dict['rewards'], cfg, tag='test'+str((i+1) * cfg['train_episodes']),
        save=True, path='./output/results/'
    )
    env_rec.close()
env.close()

# 绘制训练图像
Drawer.plot(
    train_rewards, cfg, tag='train'+str((i+1) * cfg['train_episodes']),
    save=True, path='./output/results/'
)

agent.save('./output/models/')
