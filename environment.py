import numpy as np
from params import *
import random
from itertools import count
import matplotlib.pyplot as plt
# from ROS_reference.sim_stack import hasFallen

def get_action_space(state):

    selected_piece = numpy.where(state[3:] == 1)
    place_piece = numpy.where(state[:3] == 0)
    
    if len(place_piece) < 3:
        place_piece.extend([-1, -2, -3])
    
    return [selected_piece, place_piece]

def simulate(state, action):
    
    selected_piece, place_piece = action
    increase_height = False
    is_end = False

    # if add to new level
    state[selected_piece] = 0
    if place_piece < 0:
        increase_height = True
        state = np.concatenate((np.zeros(3), state), axis=0)
        state[place_piece + 3] = 1
    else:
        state[place_piece] = 1


    # interact with ROS environment
    if hasFallen(state):
        is_end = True

    return is_end, increase_height

class JengaEnv:
    def __init__(self, args):
        self.args = args
        self.state = np.ones(self.args.init_height * 3)
        self.curr_height = 0
        self.is_end = False

    def reset(self):
        self.state = np.ones(self.args.init_height * 3)
        self.curr_height = 0
        self.is_end = False
        
        return self.state, self.curr_height, self.is_end

    def step(self, action):
        # drop the block
        self.is_end, increase_height= simulate(self.state, action)
        if increase_height:
            self.curr_height += 1

        return self.state, self.curr_height, self.is_end

# def calc_reward(rows_cleared_prev, rows_cleared, is_end, reward_type="all"):
#     if reward_type == "cleared":
#         return (rows_cleared - rows_cleared_prev) * 5
#     else:
#         if is_end:
#             reward = -10
#         else:
#             reward = 1 + (rows_cleared - rows_cleared_prev) * 5
#         return reward


def test(args):
    
    env = JengaEnv(args=args)
    state, curr_height, is_end = env.reset()
    
    for t in count():

        if is_end:
            break
        # action_space: list of indices to pick
        action_space = get_action_space(state) 
        action = [np.random.choice(action_space[0]), np.random.choice(action_space[1])]
        state, curr_height, is_end = env.step(action)