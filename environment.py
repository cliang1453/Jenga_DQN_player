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
        self.is_end = False
        
        return self.state, self.is_end

    def step(self, action):
        # drop the block
        self.state, flag = self.simulate(self.state, action)
        # is_end: 0: continue
        # 1: goal state
        # 2: fallen state
        if flag > 0: 
            self.is_end = True
        else:
            self.is_end = False

        return self.state, self.is_end

    def simulate(self, state, action):

        # what ROS did
        
        next_state = state

        to_loc = np.argwhere(next_state==1)[0]-1
        next_state[to_loc] = 1
        from_loc = np.argwhere(next_state==1)[::-1][action]
        next_state[from_loc] = 0

        if self.has_fallen(next_state):
            return next_state, 2
        elif self.reached_goal(next_state):
            return next_state, 1
        else:
            return next_state, 0

    def has_fallen(self, state):
        state_ = state.reshape((3, -1))
        idx_list = np.argwhere(state_ == 1)

        


    def reached_goal(self, state):
        if state == self.goal:
            return True
        

    def get_reward(self, state, is_end):
        
        mask = [0,1,0]
        reward = 0
        for i in range(0,len(state),3):
            if (mask==state[i:i+3]).all(): reward += 1

        if not is_end:
            if (state==self.goal).all(): reward += self.reward
            return reward
        reward += self.cost
        return reward



