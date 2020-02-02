import gym
import numpy as np
from maze2d_env import MazeEnv
from stable_baselines import DQN, PPO2, A2C, ACKTR, ACER, TD3, TRPO
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

env = MazeEnv(grid_size=5)
env = Monitor(env, filename=None, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# Train the agent
model = ACKTR('MlpPolicy', env, lr_schedule='double_linear_con',gamma=0.98, learning_rate=0.35, verbose=1,tensorboard_log="./a2c_cartpole_tensorboard/")

def evaluate(model, num_episodes=100,render=False):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            if render: env.render(mode="human")
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        # print("Episode", i,"Reward:", sum(episode_rewards))
        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    min_episode_reward = np.min(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Min reward:", min_episode_reward,"Num episodes:", num_episodes)

    return mean_episode_reward

# Test the trained agent
model.learn(500000)
# evaluate(model, num_episodes=4, render=True)
