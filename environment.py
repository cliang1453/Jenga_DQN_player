import numpy as np
import random
from itertools import count
# from ROS_reference.sim_stack import hasFallen

def get_action_space(state):

    s, curr_height = state
    max_height = int(s.shape[0]/3)

    selected_piece = np.squeeze(np.argwhere(s[-curr_height*3:] == 1), axis=1) + (max_height-curr_height)*3
    print(selected_piece)
    place_piece = np.squeeze(np.argwhere(s[(max_height - curr_height + 1) * 3 - 3:(max_height - curr_height + 1) * 3] == 0), axis=1)\
                  + (max_height - curr_height + 1) * 3 - 3


    if len(place_piece) < 3:
        new_level = np.squeeze(np.argwhere(s[(max_height - curr_height + 1) * 3 - 6: (max_height - curr_height + 1) * 3 - 3] == 0), axis=1)\
         + (max_height - curr_height + 1) * 3 - 6
        place_piece = np.append(place_piece, new_level)

    print(place_piece)
    # a list of possible actions 
    # action: [selected_piece, place_piece]
    action_space = np.array(np.meshgrid(selected_piece, place_piece), dtype=int).T.reshape(-1, 2)

    return action_space

def simulate(state, action):

    """
        Input: 
        state: field, curr_height
        action: selected_piece, place_piece

        Output: 
        next_state: next_field, next_height

    """
    next_s, next_h = state[0].copy(), state[1]
    selected_piece, place_piece = action
    is_end = False
    print(selected_piece)

    # if add to new level
    next_s[selected_piece] = 0
    if place_piece < (state[0].shape[0]/3-state[1])*3:
        next_h += 1
        next_s[place_piece] = 1
    else:
        next_s[place_piece] = 1

    # interact with ROS environment
    is_end = hasFallen(next_s, next_h)
    next_state = (next_s, next_h)

    return next_state, is_end

def hasFallen(state, curr_height):

    """
        This is a primitive fall detection serving for testing purpose! 
        TODO: add physical engine (interact with ROS)
    """

    state_T = state.reshape((-1, 3))[-curr_height:]

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
        
        s = np.zeros(self.args.init_height * 3 *3)
        s[-self.args.init_height * 3:] = 1
        self.state = (s, self.args.init_height)
        self.is_end = False
        
        return self.state, self.is_end

    def step(self, action):
        # drop the block
        self.state, self.is_end = simulate(self.state, action)

        return self.state[0], self.state[1], self.is_end

def calc_reward(prev_height, curr_height, is_end):

    if is_end:
        reward = -10
    else:
        reward = 1 + (curr_height - prev_height) * 1
        
    return reward


def test_env(args):
    
    env = JengaEnv(args=args)
    state, is_end = env.reset()
    print(state[0].reshape((-1, 3)))
    for t in count():

        if is_end:
            print(is_end)
            break
        # action_space: list of indices to pick
    # state = (np.array([[0, 1, 0],
    #                   [0, 1, 1],
    #                   [1, 1, 1],
    #                   [1, 1, 1],
    #                   [1, 1, 1],
    #                   [1, 1, 1],
    #                   [1, 1, 1],
    #                   [1, 1, 1],
    #                   [1, 1, 1],
    #                   [1, 1, 1],
    #                   [1, 1, 1]]).reshape(-1)

    
        action_space = get_action_space(state)
        
        action = action_space[np.random.choice(len(action_space))]
        s, curr_height, is_end = env.step(action)
        print(s.reshape((-1,3)))
        state = (s, curr_height)
