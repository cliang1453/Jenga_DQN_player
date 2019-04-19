import numpy as np
import random
from itertools import count
# from ROS_reference.sim_stack import hasFallen

def get_action_space(state, curr_height, max_height):
    
    selected_piece = np.squeeze(np.argwhere(state[(max_height - curr_height + 1) * 3:] == 1), axis=1) + 3
    place_piece = np.squeeze(np.argwhere(state[(max_height - curr_height + 1) * 3 - 3:(max_height - curr_height + 1) * 3] == 0), axis=1)

    if len(place_piece) < 3:
        place_piece = np.append(place_piece, state[(max_height - curr_height + 1) * 3 - 6: (max_height - curr_height + 1) * 3 - 3])

    # a list of possible actions 
    # action: [selected_piece, place_piece]
    action_space = np.array(np.meshgrid(selected_piece, place_piece)).T.reshape(-1, 2)

    return action_space

def simulate(state, action):

    """
        Input: current state, current action
        Output: next state
    """
    next_state = state.copy()

    selected_piece, place_piece = action
    increase_height = False
    is_end = False

    # if add to new level
    next_state[selected_piece] = 0
    if place_piece < 0:
        increase_height = True
        next_state = np.concatenate((np.zeros(3), next_state), axis=0)
        next_state[place_piece + 3] = 1
    else:
        next_state[place_piece] = 1

    # interact with ROS environment
    is_end = hasFallen(next_state)

    return next_state, is_end, increase_height

def hasFallen(state):

    """
        This is a primitive fall detection serving for testing purpose! 
        TODO: add physical engine (interact with ROS)
    """

    state_T = state.reshape((-1, 3))

    # check if there is all zero rows. Note that we always make sure the top row is no completely empty
    checker = np.ones((state_T.shape[0], 1))
    level_cnt = np.sum(state_T * checker, axis=1)
    if np.count_nonzero(level_cnt) < state_T.shape[0]:
        return True

    # check if there is unbalance condition (i.e. 100 or 001) and there is no support on top of it
    for idx in [0, 2]:
        unbalance_levels = np.squeeze(np.argwhere(level_cnt == 1), axis=1)
        unbalance_mask = (state_T[unbalance_levels, idx] == 1)
        unbalance_levels = np.squeeze(np.argwhere(unbalance_mask * unbalance_levels > 0), axis=1)
        
        for level in unbalance_levels:
            if level == 0:
                continue
            if level == 1:
                return True
            if state_T[level - 2][idx] == 0:
                return True

    return False


class JengaEnv(object):
    
    def __init__(self, args):
        self.args = args

    def reset(self):
        self.state = np.zeros(self.args.init_height * 3 *3)
        self.state[:self.args.init_height * 3] = 1
        self.curr_height = self.args.init_height
        self.is_end = False
        
        return self.state, self.curr_height, self.is_end

    def step(self, action):
        # drop the block
        self.state, self.is_end, increase_height = simulate(self.state, action)
        if increase_height:
            self.curr_height += 1

        return self.state, self.curr_height, self.is_end

def calc_reward(prev_height, curr_height, is_end):

    if is_end:
        reward = -10
    else:
        reward = 1 + (curr_height - prev_height) * 1
        
    return reward


def test_env(args):
    
    env = JengaEnv(args=args)
    state, curr_height, is_end = env.reset()
    
    # for t in count():

    #     if is_end:
    #         break

    #     # action_space: list of indices to pick
    state = np.array([[0, 1, 0],
                      [0, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]).reshape(-1)

    action_space = get_action_space(state, curr_height, max_height) 
    print(action_space)
    action = action_space[np.random.choice(len(action_space))]
    state, curr_height, is_end = env.step(action)
