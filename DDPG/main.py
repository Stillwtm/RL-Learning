"""
一个方便训练和进行多次实验的上层脚本
"""
import argparse
import time
from random import randint
from train import main_dqn
from utils import make_dir
from plot import draw_single_plot, draw_multi_plot

if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Pendulum-v1', help='environment name, refer to gym doc')
    parser.add_argument('--seed', nargs='+', type=int, default=None, help='random seed for all')
    parser.add_argument('--ep_max_steps', type=int, default=int(1e4), help='max steps in one episode')
    parser.add_argument('--max_train_steps', type=int, default=int(8e4), help='max training steps')
    parser.add_argument('--eval_every', type=int, default=int(500), help='evaluate interval, in steps')
    parser.add_argument('--save_model_every', type=int, default=int(5e4), help='save model interval, in steps')
    parser.add_argument('--update_every', type=int, default=50, help='update interval, in steps')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e4), help='capacity of replay buffer')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of Q net')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='soft update factor')
    parser.add_argument('--noise_sigma', type=float, default=0.01, help='sigma of gaussian noise')
    parser.add_argument('--actor_lr', type=float, default=3e-4, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='learning rate of critic')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to sample form replay buffer')
    parser.add_argument('--start_size', type=int, default=1024, help='initial buffer size to start training')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    # 脚本的一些设置，和训练关系不大
    parser.add_argument('--run', type=int, default=1, help='run the whole experiment process how many times')
    parser.add_argument('--log_path', type=str, default='./output/results/', help='dir to save log file')
    parser.add_argument('--model_path', type=str, default='./output/models/', help='dir to save model')
    opt = parser.parse_args()
    default_args = {
        'algo_name': 'DDPG',
    }
    cfg = {**vars(opt), **default_args}  # 变成字典，个人觉得方便一点

    start_time = time.time()

    cfg['log_path'] = make_dir(cfg)
    if cfg['seed'] is None:
        for i in range(cfg['run']):
            cfg['seed'] = randint(1, 1024)
            print(f"===Start training with seed {cfg['seed']}. Run:{i+1}/{cfg['run']}===")
            main_dqn(cfg)
    else:
        for i, seed in enumerate(opt.seed):
            cfg['seed'] = seed
            print(f"===Start training with seed {cfg['seed']}. Run:{i+1}/{len(opt.seed)}===")
            main_dqn(cfg)
    
    # 画图
    draw_single_plot(cfg['log_path'], plot_smooth=True, smooth_method='moving')
    draw_multi_plot(cfg['log_path'], plot_smooth=False)

    print(f"Finish running! Run all for {time.time() - start_time}s!")
