import os
import seaborn as sns
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import dill
import numpy as np

def extract_data(path):
    """是文件就返回文件，是文件夹就返回文件list
    """
    if os.path.isdir(path):
        datas = {}
        data_list = []
        files = os.listdir(path)
        for file in files:
            if os.path.splitext(file)[-1] == '.pkl':
                with open(os.path.join(path, file), 'rb') as f:
                    data = dill.load(f)
                    for k, v in data.items():
                        datas.setdefault(k, []).extend(v)
                    data_list.append(data)
    else:
        with open(path, 'rb') as f:
            datas = dill.load(f)
    return datas, data_list

def smooth(data, *args, k=10, smooth_method='savgol'):
    """平滑曲线，k为滑动窗口长度
    """
    smooth_data = data
    for arg in args:
        if len(data[arg]) > k:  # 确定是储存大量数据的再smooth
            if smooth_method == 'average':
                y = np.ones(k) * 1.0
                z = np.ones(len(data[arg]))
                smooth_data[arg] = np.convolve(y, data[arg], 'same') / np.convolve(y, z, 'same')
            elif smooth_method == 'savgol':
                smooth_data[arg] = savgol_filter(data[arg], k, 3)
            elif smooth_method == 'moving':
                last, weight = data[arg][0], 0.8
                smoothed = []
                for d in data[arg]:
                    last = last * weight + (1 - weight) * d
                    smoothed.append(last)
                smooth_data[arg] = smoothed
    return smooth_data

def draw_single_plot(data_path, plot_smooth=False, smooth_method='savgol'):
    """画单次训练数据的图像
    """
    _, data_list = extract_data(data_path)
    for data in data_list:
        plt.figure()
        plt.title(f"Curve of seed {data['seed']}")
        plt.xlabel('Epsiodes')
        plt.ylabel('Rewards')
        sns.lineplot(x='step', y='reward', data=data, label='reward')
        if plot_smooth:
            smooth_data = smooth(data, 'reward', smooth_method=smooth_method)
            sns.lineplot(x='step', y='reward', data=smooth_data, label='smooth_reward')
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(data_path), f"figure_{data['seed']}.png"))

def draw_multi_plot(data_path, plot_smooth=False, smooth_method='savgol'):
    """画所有数据的聚合图像
    """
    datas, _ = extract_data(data_path)
    plt.figure()
    plt.title(f"Curve of seed {datas['seed']}")
    plt.xlabel('Epsiodes')
    plt.ylabel('Rewards')
    sns.lineplot(x='step', y='reward', data=datas, label='reward')
    if plot_smooth:
        smooth_datas = smooth(datas, 'reward', smooth_method=smooth_method)
        sns.lineplot(x='step', y='reward', data=smooth_datas, label='smooth_reward')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(data_path), f"figure_{datas['seed']}.png"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./output/results/', help='path of data, file for single and dir for multi')
    parser.add_argument('--draw', type=str, default='multi', help='single or multi or both')
    parser.add_argument('--smooth', action='store_true', default=False, help='draw smoothed plot')
    parser.add_argument('--smooth_method', type=str, default='savgol', help='savgol or average or moving')
    opt = parser.parse_args()
    if opt.draw == 'single' or opt.draw == 'both':
        draw_single_plot(opt.path, plot_smooth=opt.smooth, smooth_method=opt.smooth_method)
    if opt.draw == 'multi' or opt.draw == 'both':
        draw_multi_plot(opt.path, plot_smooth=opt.smooth, smooth_method=opt.smooth_method)
