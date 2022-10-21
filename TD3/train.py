import gym
from td3 import TD3
from utils import Logger, NormalizedEnv, all_seed

def eval_policy(eval_env, agent, cfg, eval_episodes=3):
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

def train(env, eval_env, agent, cfg, logger):
    steps = 0
    while steps < cfg['max_train_steps']:
        state = env.reset()[0]
        for _ in range(cfg['ep_max_steps']):
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, terminated)
            state = next_state
            steps += 1
            if steps > cfg['start_size']:
                agent.update()
            if steps % cfg['eval_every'] == 0:
                score = eval_policy(eval_env, agent, cfg)
                logger.log(reward=score, step=steps)
                print(f"Train step {steps}/{cfg['max_train_steps']}: reward: {score}")
            if steps % cfg['save_model_every'] == 0:
                agent.save(cfg, steps)
            if terminated or truncated:
                break
    
    agent.save(cfg, steps)
    env.close()

def main_dqn(cfg):
    # 设置随机种子
    all_seed(seed=cfg['seed'])
    
    # 创建环境
    env = NormalizedEnv(gym.make(cfg['env_name']))
    eval_env = NormalizedEnv(gym.make(cfg['env_name']))
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
    agent = TD3(state_dim, action_dim, cfg)

    # 训练
    logger = Logger()
    logger.log(seed=cfg['seed'])
    train(env, eval_env, agent, cfg, logger)
    
    # 写入数据
    logger.dump_to_file(cfg['log_path'])