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
        
        state_ = state.reshape((-1, 3))
        idx_list = np.argwhere(state_ == 1)

        level_pos = []
        last_y = state_.shape[0]

        for i in range(idx_list.shape[0]-1, -1, -1):
            y, x = idx_list[i]

            if y != last_y:


                if y < last_y-1:
                    return True

                level_pos.append([])
                last_y = y



            if y%2 == 0:
                level_pos[-1].append([x, 1])
            else:
                level_pos[-1].append([1, x])

        level_mass = []
        level_block_cnt = []
        level_mass_centroid =[]
        
        for level in level_pos[::-1]:
            
            if len(level_mass)==0:

                level_mass.append(np.sum(np.array(level), axis=0))
                level_block_cnt.append(len(level))
            else:

                level_mass.append(level_mass[-1] + np.sum(np.array(level), axis=0))
                level_block_cnt.append(level_block_cnt[-1] + len(level))

        for i in range(len(level_block_cnt)):
            level_mass_centroid.append(level_mass[i]/level_block_cnt[i])

        for i in range(len(level_mass_centroid)-1):
            cy, cx = level_mass_centroid[i]
            dy, dx = level_mass_centroid[i+1]

            if abs(dy-cy) > 1 or abs(cx-dx) > 1:
                return True

        return False


    def reached_goal(self, state):
        if (state == self.goal).all():
            return True
        return False
        

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


def test_env(args):
    
    env = JengaEnv(args)
    state, is_end = env.reset()

    for t in count():

        if is_end:
            break

        print(state.reshape(-1, 3))

        action = np.random.choice(np.argwhere(state==1).squeeze())
        print(action)
        state, is_end = env.step(action)
