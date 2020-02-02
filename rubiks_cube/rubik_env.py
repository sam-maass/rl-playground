import numpy as np
import random
import gym
from gym import spaces


class RubikEnv(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where the agent must learn to go always left. 
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['console']}
  # Define constants for clearer code
  TURN_LEFT = 0
  TURN_RIGHT = 1
  TURN_UP = 2
  TURN_DOWN = 3
  ROTATE_LEFT_COLUMN_UP = 4
  ROTATE_LEFT_COLUMN_DOWN = 5
  ROTATE_RIGHT_COLUMN_UP = 6
  ROTATE_RIHT_COLUMN_DOWN = 7
  ROTATE_UPPER_ROW_LEFT = 8
  ROTATE_UPPER_ROW_RIGHT = 9
  ROTATE_LOWER_ROW_LEFT = 10
  ROTATE_LOWER_ROW_RIGHT = 11

  RED = 1
  YELLOW = 2
  BLUE = 3
  GREEN = 4
  WHITE = 5
  BLACK = 6

  def get_cube_side(self,color):
    return [[color,color,color],[color,color,color],[color,color,color]]

  def get_cube(self):
    cube = {
      "front": self.get_cube_side(self.RED),
      "top": self.get_cube_side(self.YELLOW),
      "back": self.get_cube_side(self.BLUE),
      "bottom": self.get_cube_side(self.GREEN),
      "left": self.get_cube_side(self.WHITE),
      "right": self.get_cube_side(self.BLACK)
    }
    return cube
  
  def turn_left(self):
    temp = self.cube["front"]
    self.cube["front"] = self.cube["right"]
    self.cube["right"] = self.cube["back"]
    self.cube["back"] = self.cube["left"]
    self.cube["left"] = temp
  
  def turn_right(self):
    temp = self.cube["front"]
    self.cube["front"] = self.cube["left"]
    self.cube["left"] = self.cube["back"]
    self.cube["back"] = self.cube["right"]
    self.cube["right"] = temp
  
  def turn_up(self):
    temp = self.cube["front"]
    self.cube["front"] = self.cube["bottom"]
    self.cube["bottom"] = self.cube["back"]
    self.cube["back"] = self.cube["top"]
    self.cube["top"] = temp
  
  def turn_down(self):
    temp = self.cube["front"]
    self.cube["front"] = self.cube["top"]
    self.cube["top"] = self.cube["back"]
    self.cube["back"] = self.cube["bottom"]
    self.cube["bottom"] = temp

  def rotate_left_column_down(self):
    temp = [self.cube["front"][0][0],self.cube["front"][1][0],self.cube["front"][2][0]]
    for i in range(3):
      self.cube["front"][i][0] = self.cube["top"][i][0]
    for i in range(3):
      self.cube["top"][i][0] = self.cube["back"][i][0]
    for i in range(3):
      self.cube["back"][i][0] = self.cube["bottom"][i][0]
    for i in range(3):
      self.cube["bottom"][i][0] = temp[i]

  def rotate_left_column_up(self):
    temp = [self.cube["front"][0][0],self.cube["front"][1][0],self.cube["front"][2][0]]
    for i in range(3):
      self.cube["front"][i][0] = self.cube["bottom"][i][0]
    for i in range(3):
      self.cube["bottom"][i][0] = self.cube["back"][i][0]
    for i in range(3):
      self.cube["back"][i][0] = self.cube["top"][i][0]
    for i in range(3):
      self.cube["top"][i][0] = temp[i]

  def rotate_right_column_down(self):
    temp = [self.cube["front"][0][2],self.cube["front"][1][2],self.cube["front"][2][2]]
    for i in range(3):
      self.cube["front"][i][2] = self.cube["top"][i][2]
    for i in range(3):
      self.cube["top"][i][2] = self.cube["back"][i][2]
    for i in range(3):
      self.cube["back"][i][2] = self.cube["bottom"][i][2]
    for i in range(3):
      self.cube["bottom"][i][2] = temp[i]

    def rotate_right_column_up(self):
      temp = [self.cube["front"][0][2],self.cube["front"][1][2],self.cube["front"][2][2]]
      for i in range(3):
        self.cube["front"][i][2] = self.cube["bottom"][i][2]
      for i in range(3):
        self.cube["bottom"][i][2] = self.cube["back"][i][2]
      for i in range(3):
        self.cube["back"][i][2] = self.cube["top"][i][2]
      for i in range(3):
        self.cube["top"][i][2] = temp[i]

  def rotate_right_column_up(self):
    temp = [self.cube["front"][0][2],self.cube["front"][1][2],self.cube["front"][2][2]]
    for i in range(3):
      self.cube["front"][i][2] = self.cube["bottom"][i][2]
    for i in range(3):
      self.cube["bottom"][i][2] = self.cube["back"][i][2]
    for i in range(3):
      self.cube["back"][i][2] = self.cube["top"][i][2]
    for i in range(3):
      self.cube["top"][i][2] = temp[i]
      
  def rotate_top_row_left(self):
    temp = self.cube["front"][0]
    self.cube["front"][0] = self.cube["right"][0]
    self.cube["right"][0] = self.cube["back"][0]
    self.cube["back"][0] = self.cube["left"][0]
    self.cube["left"][0] = temp
      
  def rotate_top_row_right(self):
    temp = self.cube["front"][0]
    self.cube["front"][0] = self.cube["left"][0]
    self.cube["left"][0] = self.cube["back"][0]
    self.cube["back"][0] = self.cube["right"][0]
    self.cube["right"][0] = temp
      
  def rotate_bottom_row_left(self):
    temp = self.cube["front"][2]
    self.cube["front"][2] = self.cube["right"][2]
    self.cube["right"][2] = self.cube["back"][2]
    self.cube["back"][2] = self.cube["left"][2]
    self.cube["left"][2] = temp
      
  def rotate_bottom_row_right(self):
    temp = self.cube["front"][2]
    self.cube["front"][2] = self.cube["left"][2]
    self.cube["left"][2] = self.cube["back"][2]
    self.cube["back"][2] = self.cube["right"][2]
    self.cube["right"][2] = temp

  def twist_cube(self):
    actions = [self.turn_down,self.turn_left,self.rotate_top_row_left,self.rotate_bottom_row_right]
    for turn in range(2):
      index = random.randint(0,3)
      actions[index]()


  def __init__(self, grid_size=10):
    super(RubikEnv, self).__init__()
    self.cube = self.get_cube()
    self.twist_cube()
    self.step_count = 0

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions, we have two: left and right
    n_actions = 12
    self.action_space = spaces.Discrete(n_actions)

    print(np.array(self.cube['front']).flatten())

    self.observation_space = spaces.Box(low=0, high=5,
                                        shape=(3,3), dtype=np.int)

  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    self.cube = self.get_cube()
    self.step_count = 0
    self.twist_cube()
    return np.array(self.cube['front'])

  def step(self, action):
    if action == self.ROTATE_LEFT_COLUMN_DOWN:
      self.rotate_left_column_down()
    elif action == self.ROTATE_LEFT_COLUMN_UP:
      self.rotate_left_column_up()
    elif action == self.ROTATE_RIHT_COLUMN_DOWN:
      self.rotate_right_column_down()
    elif action == self.ROTATE_RIGHT_COLUMN_UP:
      self.rotate_right_column_up()
    elif action == self.ROTATE_LOWER_ROW_LEFT:
      self.rotate_bottom_row_left()
    elif action == self.ROTATE_LOWER_ROW_RIGHT:
      self.rotate_bottom_row_right()
    elif action == self.ROTATE_UPPER_ROW_LEFT:
      self.rotate_top_row_left()
    elif action == self.ROTATE_UPPER_ROW_RIGHT:
      self.rotate_top_row_right()
    elif action == self.TURN_DOWN:
      self.turn_down()
    elif action == self.TURN_UP:
      self.turn_up()
    elif action == self.TURN_LEFT:
      self.turn_left()
    elif action == self.TURN_RIGHT:
      self.turn_right()
    else:
      pass
      # raise ValueError("Received invalid action={} which is not part of the action space".format(action))

    self.step_count += 1
    reward = -0.2
    correct_items_on_front = 0
    for rows in self.cube["front"]:
      for item in rows:
        if item == self.RED:
         correct_items_on_front += 1
         reward = + 0.1/9
    
    done = correct_items_on_front == 9 or self.step_count > 1000
    if done:
      reward += 100
    info = {}

    return np.array(self.cube["front"]), reward, done, info

  def render(self, mode='console'):
    print(self.cube["front"])

  def close(self):
    pass
    