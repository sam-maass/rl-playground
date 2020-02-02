import numpy as np
import gym
from gym import spaces
import random


class MazeEnv(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where the agent must learn to go always left. 
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['console']}
  # Define constants for clearer code
  LEFT = 0
  RIGHT = 1
  UP = 2
  DOWN = 3

  def __init__(self, grid_size=10):
    super(MazeEnv, self).__init__()

    # Size of the 1D-grid
    self.grid_size = grid_size
    # Initialize the agent at the right of the grid
    self.agent_pos_x = 0
    self.agent_pos_y = 0
    self.agent_steps =0

    self.exit_pos_x = random.randint(0,grid_size - 1)
    self.exit_pos_y = random.randint(0,grid_size - 1)

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions, we have two: left and right
    n_actions = 4
    self.action_space = spaces.Discrete(n_actions)
    # The observation will be the coordinate of the agent
    # this can be described both by Discrete and Box space
    self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                        shape=(3,), dtype=np.float32)

  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    # Initialize the agent at the right of the grid
    self.agent_pos_x = 0
    self.agent_pos_y = 0
    self.agent_steps = 0

    self.exit_pos_x = random.randint(0,self.grid_size - 1)
    self.exit_pos_y = random.randint(0,self.grid_size - 1)

    # here we convert to float32 to make it more general (in case we want to use continuous actions)
    return np.array([self.agent_pos_x,self.agent_pos_y,self.grid_size]).astype(np.float32)

  def step(self, action):
    self.agent_steps += 1
    if action == self.LEFT:
      self.agent_pos_x -= 1
    elif action == self.RIGHT:
      self.agent_pos_x += 1
    elif action == self.UP:
      self.agent_pos_y += 1
    elif action == self.DOWN:
      self.agent_pos_y -= 1
    else:
      raise ValueError("Received invalid action={} which is not part of the action space".format(action))
    
    old_agent_pos_x = self.agent_pos_x
    old_agent_pos_y = self.agent_pos_y

    # Account for the boundaries of the grid
    self.agent_pos_x = np.clip(self.agent_pos_x, 0, self.grid_size-1)
    self.agent_pos_y = np.clip(self.agent_pos_y, 0, self.grid_size-1)

    # Are we at the left of the grid?
    done = self.agent_pos_x == self.exit_pos_x and self.agent_pos_y == self.exit_pos_y 

    # Null reward everywhere except when reaching the goal (left of the grid)
    reward = -1/(self.grid_size*self.grid_size)
    if done:
      reward = 1
    elif (old_agent_pos_x != self.agent_pos_x or old_agent_pos_y != self.agent_pos_y):
      reward = -1


    # Optionally we can pass additional info, we are not using that for now
    info = {}

    done = done or self.agent_steps >= 500

    return np.array([self.agent_pos_x,self.agent_pos_y,self.grid_size]).astype(np.float32), reward, done, info

  def render(self, mode='console'):
    array = np.zeros((self.grid_size,self.grid_size))
    array[self.agent_pos_x][self.agent_pos_y] = 1
    array[self.exit_pos_x][self.exit_pos_y] = 2

    print(np.matrix(array))
    print("\n")

  def close(self):
    pass
    