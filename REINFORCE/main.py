from solver import Solver, Drawer

cfg = {
    'algo_name': 'REINFORCE',
    'env_name': 'CartPole-v1',
    'train_episodes': 200,
    'test_episodes': 20,
    'ep_max_steps': 100000,
    'lr': 0.01,
    'gamma': 0.99,
    'device': 'cpu',
}

env, agent = Solver.create_env_agent(cfg)
# env = env.unwrapped
train_rewards = []
for i in range(5):
    # 训练
    res_dict = Solver.train(env, agent, cfg)
    train_rewards += res_dict['rewards']
    # 测试+录像
    env_rec = Solver.create_rec_env(cfg, tag=str((i+1) * cfg['train_episodes']))
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

# agent.save('./output/models/')