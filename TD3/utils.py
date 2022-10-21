import datetime
import os
import dill
import gym

def all_seed(seed=1):
    import torch
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def make_dir(cfg, new_dir=True):
    """创建储存结果的文件夹
        new_dir: True就按照时间创建新文件夹, False就不创建
    """
    dir_path = cfg['log_path']
    if new_dir:
        cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_path = os.path.join(cfg['log_path'], f"{cfg['algo_name']}_{cfg['env_name']}_{cur_time}/")
        os.mkdir(dir_path)
    elif not os.path.exists(cfg['log_path']):
        os.mkdir(dir_path)
    return dir_path

class Logger(object):
    def __init__(self):
        self.storage = {}
    
    def log(self, **args):
        for k, v in args.items():
            self.storage.setdefault(k, []).append(v)
    
    def dump_to_file(self, dir_path):
        log_path = dir_path + f"data_seed{self.storage['seed'][0]}.pkl"
        with open(log_path, 'wb') as f:
            dill.dump(self.storage, f)
        return log_path

class NormalizedEnv(gym.ActionWrapper):
    """[-1, 1] -> [low, high]
    参见https://github.com/openai/gym/blob/master/gym/core.py
    """
    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)