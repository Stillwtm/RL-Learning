import gym
import argparse
from gym.wrappers.record_video import RecordVideo
from dqn import DQN
from utils import Logger, all_seed, make_dir
from plot import draw_plot

# 超参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='CartPole-v1', help='environment name, refer to gym doc')
parser.add_argument('--seed', type=int, default=40, help='random seed for all')
parser.add_argument('--ep_max_steps', type=int, default=int(1e4), help='max steps in one episode')
parser.add_argument('--max_train_steps', type=int, default=int(8e4), help='max training steps')
parser.add_argument('--eval_every', type=int, default=int(500), help='evaluate interval, in steps')
parser.add_argument('--save_model_every', type=int, default=int(5e4), help='save model interval, in steps')
parser.add_argument('--update_every', type=int, default=50, help='update interval, in steps')
parser.add_argument('--buffer_capacity', type=int, default=int(1e4), help='capacity of replay buffer')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of Q net')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epsilon_start', type=float, default=0.95, help="initial value of epsilon")
parser.add_argument('--epsilon_end', type=float, default=0.01, help="final value of epsilon")
parser.add_argument('--epsilon_decay', type=int, default=500, help="decay rate of epsilon, the higher value, the slower decay")
parser.add_argument('--batch_size', type=int, default=64, help='batch size to sample form replay buffer')
parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
parser.add_argument('--no_auto_dir', action='store_true', default=False, help='do not automatically create new directory to save data')
parser.add_argument('--log_path', type=str, default='./output/results/', help='dir to save log file')
parser.add_argument('--model_path', type=str, default='./output/models/', help='dir to save model')
opt = parser.parse_args()
default_args = {
    'algo_name': 'DQN',
}
cfg = {**vars(opt), **default_args}  # 变成字典，个人觉得方便一点

def eval_policy(eval_env, agent, eval_episodes=3):
    scores = 0
    for _ in range(eval_episodes):
        state = eval_env.reset()[0]
        for _ in range(cfg['ep_max_steps']):
            action = agent.predict_action(state)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            state = next_state
            scores += reward
            if terminated or truncated:
                break
    eval_env.close()
    return scores / eval_episodes

def train(env, eval_env, agent, logger):
    steps = 0
    while steps < cfg['max_train_steps']:
        state = env.reset()[0]
        for _ in range(cfg['ep_max_steps']):
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, terminated or truncated)
            state = next_state
            steps += 1
            agent.update()
            if steps % cfg['eval_every'] == 0:
                score = eval_policy(eval_env, agent)
                logger.log(reward=score, step=steps)
                print(f"Train step {steps}/{cfg['max_train_steps']}: reward: {score}")
            if steps % cfg['save_model_every'] == 0:
                agent.save(cfg, steps)
            if terminated or truncated:
                break
    
    agent.save(cfg, steps)
    env.close()

def main_dqn():
    # 设置随机种子
    all_seed(seed=cfg['seed'])
    
    # 创建环境
    env = gym.make(cfg['env_name'])
    eval_env = gym.make(cfg['env_name'])
    env.reset(seed=cfg['seed'])
    eval_env.reset(seed=cfg['seed'])
    try:
        state_dim = env.observation_space.n  # 离散
    except AttributeError:
        state_dim = env.observation_space.shape[0]  # 连续
    try:
        action_dim = env.action_space.n  # 离散
    except AttributeError:
        action_dim = env.action_space.shape[0]  # 连续
    
    # 创建智能体
    agent = DQN(state_dim, action_dim, cfg)

    # 训练
    logger = Logger()
    logger.log(seed=cfg['seed'])
    train(env, eval_env, agent, logger)
    
    # 写入数据
    dir_path = make_dir(cfg, new_dir=not cfg['no_auto_dir'])
    log_path = logger.dump_to_file(dir_path)

    # 画图
    draw_plot(log_path, plot_smooth=True)

if __name__ == '__main__':
    main()
