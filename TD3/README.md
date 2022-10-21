parser = argparse.ArgumentParser()

​    parser.add_argument('--env_name', type=str, default='Pendulum-v1', help='environment name, refer to gym doc')

​    parser.add_argument('--seed', nargs='+', type=int, default=None, help='random seed for all')

​    parser.add_argument('--ep_max_steps', type=int, default=int(1e4), help='max steps in one episode')

​    parser.add_argument('--max_train_steps', type=int, default=int(8e4), help='max training steps')

​    parser.add_argument('--eval_every', type=int, default=int(500), help='evaluate interval, in steps')

​    parser.add_argument('--save_model_every', type=int, default=int(5e4), help='save model interval, in steps')

​    parser.add_argument('--update_every', type=int, default=50, help='update interval, in steps')

​    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')

​    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension of Q net')

​    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')

​    parser.add_argument('--tau', type=float, default=0.005, help='soft update factor')

​    parser.add_argument('--noise_sigma', type=float, default=0.1, help='sigma of gaussian noise for exploration')

​    parser.add_argument('--policy_noise', type=float, default=0.2, help='noise added to target policy during critic update')

​    parser.add_argument('--noise_clip', type=float, default=0.5, help='range to clip target policy noise')

​    parser.add_argument('--policy_freq', type=int, default=2, help='frequency of delayed policy updates')

​    parser.add_argument('--actor_lr', type=float, default=3e-4, help='learning rate of actor')

​    parser.add_argument('--critic_lr', type=float, default=3e-4, help='learning rate of critic')

​    parser.add_argument('--batch_size', type=int, default=256, help='batch size to sample form replay buffer')

​    parser.add_argument('--start_size', type=int, default=1024, help='initial buffer size to start training')

​    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')

​    \# 脚本的一些设置，和训练关系不大

​    parser.add_argument('--run', type=int, default=1, help='run the whole experiment process how many times')

​    parser.add_argument('--log_path', type=str, default='./output/results/', help='dir to save log file')

​    parser.add_argument('--model_path', type=str, default='./output/models/', help='dir to save model')