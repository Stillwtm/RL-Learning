from solver import Solver, Drawer

cfg = {
    'algo_name': 'REINFORCE',
    'env_name': 'Pendulum-v1', #'CartPole-v1',
    'continuous_act': False,
    'train_episodes': 100,
    'test_episodes': 20,
    'ep_max_steps': 100000,
    'actor_lr': 0.0005,
    'critic_lr': 0.001,
    'coef_entropy': 0.001,
    'clip_eps': 0.2,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'steps_per_batch': 100,
    'run_epochs': 5,
    'batch_size': 50,
    'hidden_dim': 256,
    'device': 'cpu',
}

env, agent = Solver.create_env_agent(cfg)
train_rewards = []
last_steps = 0  # 记录前一轮训练已经跑了多少step，防止前一轮训练剩下来的没有记录
for i in range(5):
    # 训练
    res_dict = Solver.train(env, agent, cfg, last_steps)
    train_rewards += res_dict['rewards']
    last_steps = res_dict['last_steps']
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

agent.save('./output/models/')