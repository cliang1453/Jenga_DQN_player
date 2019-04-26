#!/usr/bin/env python

import numpy as np
from itertools import count

class JengaEnv(object):
    
    def __init__(self, args):
        self.height = args.init_height
        self.goal = np.tile([0,1,0],self.height*3-2)
        self.goal[[0,2]] = 1
        self.reward = 10
        self.cost = -10

    def reset(self):
        self.state = np.zeros((self.height*3-2)*3)
        self.state[-self.height*3:] = 1
        self.is_end = 0
        
        return self.state, self.is_end

    def step(self, action):
        # drop the block
        self.state, self.is_end = self.simulate(self.state, action)
        # is_end: 0: continue
        # 1: goal state
        # 2: fallen state
        return self.state, self.is_end

    def simulate(self, state, action):

        # what ROS did

        next_state = state
        to_loc = np.argwhere(next_state==1)[0][0]-1

        if to_loc == -1:
            return next_state, 2

        next_state[to_loc] = 1
        from_loc = action
        next_state[from_loc] = 0

        if self.has_fallen(next_state):
            return next_state, 2
        elif self.reached_goal(next_state):
            return next_state, 1
        else:
            return next_state, 0
    
    def has_fallen(self, state):
        
        state_ = state.reshape(-1, 3)
        # print(state_)

        # idx_list: the list of index that are blocks
        # larger y -> bottom
        # smaller y -> top
        idx_list = np.argwhere(state_ == 1)

        # level_pos: [height of tower, number of blocks at current level] (indexing)
        # larger index -> top
        # smaller index -> bottom
        level_pos = []
        last_y = state_.shape[0]


        # loop through bottom to top of the tower
        for i in range(idx_list.shape[0]-1, -1, -1):
            y, x = idx_list[i]

            if y != last_y:

                if y < last_y-1:
                    return True

                level_pos.append([y])
                last_y = y

            if y%2 == 0:
                level_pos[-1].append([x, 1])
            else:
                level_pos[-1].append([1, x])

        # the prefix sum of the mass higher or equal to current level of the tower
        # larger index -> bottom
        # smaller index -> top
        level_mass = []

        # the prefix sum of the number of blocks higher or equal to current level of the tower
        # larger index -> bottom
        # smaller index -> top
        level_block_cnt = []

        # the mass centrod position of all blocks higher or equal to current level of the tower
        # larger index -> bottom
        # smaller index -> top
        level_mass_centroid =[]
        
        # loop through top to bottom of the tower
        for level in level_pos[::-1]:
            
            if len(level_mass)==0:

                level_mass.append(np.sum(np.array(level[1:]), axis=0))
                level_block_cnt.append(len(level[1:]))
            else:

                level_mass.append(level_mass[-1] + np.sum(np.array(level[1:]), axis=0))
                level_block_cnt.append(level_block_cnt[-1] + len(level[1:]))

        for i in range(len(level_block_cnt)):
            level_mass_centroid.append(level_mass[i]/level_block_cnt[i])


        for i in range(len(level_pos)-1):
            
            cx, cy = level_mass_centroid[i]
            level = level_pos[::-1][i+1]

            # CASE I: the centroid above the current level lying on one block of current level
            if level[0]%2 == 0:
                if cx > np.amax(np.array(level[1:]), axis=0)[0] + 0.5 or cx < np.amin(np.array(level[1:]), axis=0)[0] - 0.5:
                    return True
            else:
                if cy > np.amax(np.array(level[1:]), axis=0)[1] + 0.5 or cy < np.amin(np.array(level[1:]), axis=0)[1] - 0.5:
                    return True

        return False


    def reached_goal(self, state):
        if (state == self.goal).all():
            return True
        return False
        

    def get_reward(self, state, is_end, t):

        mask_pos = [0,1,0]
        mask_neg = [1,0,1]
        reward = t

        if is_end == 2:
           return self.cost
        if (state==self.goal).all():
           return self.reward
        for i in range(0,len(state),3):
           if (mask_pos==state[i:i+3]).all(): reward += 2
           if (mask_neg==state[i:i+3]).all(): reward -= 2
        return reward

        
        return reward


def test_env(args):
    
    env = JengaEnv(args)
    state, is_end = env.reset()

    for t in count():

        if is_end:
            break

        action = np.random.choice(np.argwhere(state==1).squeeze())
        state, is_end = env.step(action)
