import os
from random import seed
import seaborn as sns
import matplotlib.pyplot as plt
import dill
import numpy as np

def extract_data(path):
    """是文件就返回文件，是文件夹就返回文件list
    """
    if os.path.isdir(path):
        datas = {}
        files = os.listdir(path)
        print(files)
        for file in files:
            if os.path.splitext(file)[-1] == '.pkl':
                with open(os.path.join(path, file), 'rb') as f:
                    data = dill.load(f)
                    for k, v in data.items():
                        datas.setdefault(k, []).extend(v)
    else:
        with open(path, 'rb') as f:
            datas = dill.load(f)
    return datas

def smooth(data, *args, k=10):
    """平滑曲线，k为滑动窗口长度
    """
    smooth_data = data
    for arg in args:
        if len(data[arg]) > k:  # 确定是储存大量数据的再smooth
            y = np.ones(k) * 1.0
            z = np.ones(len(data[arg]))
            smooth_data[arg] = np.convolve(y, data[arg], 'same') / np.convolve(z, data[arg], 'same')
    return smooth_data

def draw_plot(data_path, plot_smooth=False):
    data = extract_data(data_path)
    plt.figure()
    plt.title(f"Curve of seed {data['seed']}")
    plt.xlabel('Epsiodes')
    plt.ylabel('Rewards')
    sns.lineplot(x='step', y='reward', data=data, label='reward')
    if plot_smooth:
        smooth_data = smooth(data, 'reward')
        sns.lineplot(x='step', y='reward', data=smooth_data, label='smooth_reward')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(data_path), f"figure_{data['seed']}.png"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./output/results/', help='path of data, file for single and dir for multi')
    opt = parser.parse_args()
    draw_plot(opt.path)
