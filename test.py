import gym
import gym_maze
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


env = gym.make('gym-maze')
# vectorized environments allow to easily multiprocess training
# we demonstrate its usefulness in the next examples
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=0)

def evaluate(model, num_episodes=100):
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
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        print("Episode", i,"Reward:", sum(episode_rewards))
        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    min_episode_reward = np.min(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Min reward:", min_episode_reward,"Num episodes:", num_episodes)

    return mean_episode_reward

# Random Agent, before training
mean_reward_before_train = evaluate(model, num_episodes=100)

# Train the agent for 10000 steps
model.learn(total_timesteps=100000)

# Evaluate the trained agent
mean_reward = evaluate(model, num_episodes=100)
