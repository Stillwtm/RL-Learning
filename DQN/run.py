"""
随机取多个种子，跑多次代码，方便画在一张图上
"""
import os
from random import randint
import datetime

num_seeds = 10
cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dir_path = os.path.join('./output/results/', f"run_{num_seeds}_seeds_{cur_time}/")
for _ in range(num_seeds):
    seed = randint(1, 1024)
    os.system(f"python main.py --seed {seed} --no_auto_dir --log_path {dir_path}")