import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import random
from termcolor import colored
from collections import deque 
import pickle

from environment import *
from model import *
import torch
import torch.optim as optim
from torch.autograd import Variable


def parse_args():
    
    parser = argparse.ArgumentParser("Jenga DQN player")
    parser.add_argument("--init-height", type=int, default=10, help="the initial height of tower")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--alpha", type=float, default=0.95, help="alpha")
    parser.add_argument("--eps", type=float, default=0.01, help="eps")
    parser.add_argument("--epsilon-g", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes")
    parser.add_argument("--num-games", type=int, default=10, help="number of games")
    parser.add_argument("--batch-size", type=int, default=128, help="number of games")
    parser.add_argument("--save-dir", type=str, default="log", help="directory to save policy and plots")
    parser.add_argument("--save-interval", type=int, default=10, help="interval to save validation plots")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--max-capacity", type=int, default=50000, help="maximum capacity of replay buffer")
    parser.add_argument("--learning-start", type=int, default=100, help="learning start after number of episodes")
    parser.add_argument("--num_target_update_iter", type=int, default=1, help="number of iterations to update target Q")
    parser.add_argument("--sample_t", type=bool, default=False, help="sample t by max")

    args = parser.parse_args()
    return args



class Policy:
    
    def __init__(self, args):
        
        self.args = args
        self.logger = args.logger
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.q_func = QFunc(args).type(self.dtype)
        self.target_q_func = QFunc(args).type(self.dtype)
        self.optimizer = optim.RMSprop(self.q_func.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)

    def to_variable(self, array_in):
        if torch.cuda.is_available():
            return Variable(torch.from_numpy(array_in)).float().cuda()
        else:
            return Variable(torch.from_numpy(array_in)).float()

    def get_data(self, var_in):
        if torch.cuda.is_available():
            return var_in.data.cpu().numpy()
        else:
            return var_in.data.numpy()

    def take_action(self, state, strategy="epsilon_greedy"):

        """
        	Input: state: env.state
        	Ouput: max_action: the idx of the action gives max_Q for current state 

        """

        # epsilon greedy
        if (strategy == "epsilon_greedy" and random.random() < self.args.epsilon_g) or strategy == 'random':
            action = np.random.choice(np.argwhere(state==1).squeeze())
            return action

        # if strategy == 'validation':
        #     print(state)
        
        qs = self.get_Qs(state, q_func=self.q_func).detach()
        qs = self.get_data(qs)

        # if strategy == 'validation':
        #     print(qs)

        max_action = np.argmax(qs)
        return max_action

    def get_mask(self, state):
        
        i = 0
        # Locate the top layer of the tower
        while i < len(state) and np.sum(state[i:i+3]) == 0:
            i += 3
        # Break if unable to locate
        if i >= len(state):
            return None, None
        # If the top layer is filled, then we can choose any piece other than the top layer
        if np.sum(state[i:i + 3]) == 3:
            i += 3
        # Otherwise, we can choose any piece other than the top two layers
        else:
            i += 6
        
        mask = np.array(state)
        mask[:i] = 0
        return mask

    def get_direction_info(self, state):
        
        """
            input: state [len(state_space)/3, 3]
            output: dir_map [len(state_space)/3, 3]
        """

        h = state.shape[0]
        dir_map = np.zeros([h, 3])
        if h%2 == 0:
            dir1_level = np.arange(0, h, 2)
        else:
            dir1_level = np.arange(1, h, 2)

        dir_map[dir1_level] = 1
        return dir_map


    def get_Qs(self, states, q_func, single = True):

        state_list = []
        state_mask_list = []
        state_valididx_list = []

        if single:
            states = np.expand_dims(states, axis=0)
 
        for state in states:

            state_ = state.reshape(-1, 3) #[len(state_space)/3, 3]

            # state_mask: 1: valid index, 0: invalid index 
            state_mask = self.get_mask(state) #[len(state_space)]

            state_ = np.stack([state_, self.get_direction_info(state_)]) #[2, len(state_space)/3, 3]
            state_ = np.expand_dims(state_, axis=0).astype(float) #[1, 2, len(state_space)/3, 3]
            
            state_list.append(state_)
            state_mask_list.append(state_mask)

        states_ = np.vstack(state_list) #[batch_size, 2, len(state_space)/3, 3]
        state_masks = np.vstack(state_mask_list) #[batch_size, len(state_space)]
        states_var = self.to_variable(states_)
        Qs = q_func(states_var) # [batch_size, len(state_space)]

        # renormalization
        if single:
            
            # print(Qs)
            # print(torch.tensor(state_masks, dtype=torch.float).to(torch.device("cuda")))
            # print(Qs * torch.tensor(state_masks, dtype=torch.float).to(torch.device("cuda")))
            # print(torch.sum(Qs * torch.tensor(state_masks, dtype=torch.float).to(torch.device("cuda"))))
            # print(Qs/torch.sum(Qs * torch.tensor(state_masks, dtype=torch.float).to(torch.device("cuda"))))

            norm_Q = Qs/torch.sum(Qs * torch.tensor(state_masks, dtype=torch.float).to(torch.device("cuda"))) # [batch_size, len(state_space)]
        else:

            # print(Qs.shape)
            # print(torch.tensor(state_masks, dtype=torch.float).to(torch.device("cuda")).shape)
            # print(Qs * torch.tensor(state_masks, dtype=torch.float).to(torch.device("cuda")).shape)
            # print(torch.sum(Qs * torch.tensor(state_masks, dtype=torch.float).to(torch.device("cuda")), dim=1).shape)

            norm_Q = Qs/torch.unsqueeze(torch.sum(Qs * torch.tensor(state_masks, dtype=torch.float).to(torch.device("cuda")), dim=1), dim=1) # [batch_size, len(state_space)]

            # print(norm_Q.shape)

        norm_masked_Q = norm_Q * torch.tensor(state_masks, dtype=torch.float).to(torch.device("cuda")) # [batch_size, len(state_space)]

        return norm_masked_Q

    def learn(self, samples):

        # compute ys
        states, actions, next_states, rewards, is_ends = samples
        ys = []

        for i in range(self.args.batch_size):
            if is_ends[i] > 0:
                ys.append(rewards[i])
            else:
                qs = self.get_Qs(next_states[i], q_func=self.target_q_func).detach()
                max_q = np.asscalar(np.amax(self.get_data(qs)))
                ys.append(rewards[i] + self.args.gamma * max_q)

        # compute Q
        loss = 0
        Qs = self.get_Qs(states, single=False, q_func=self.q_func)
        ys_var = self.to_variable(np.array(ys)).view(-1, 1)
        loss = torch.sum((Qs - ys_var) ** 2) / self.args.batch_size

        ave_q = np.average(self.get_data(Qs))
        self.args.logger.add_q(ave_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self.get_data(loss).item()

    def update_target_q(self):
        self.target_q_func.load_state_dict(self.q_func.state_dict())

    def save_params(self, episode=None):
        
        filename = os.path.join(self.args.save_dir)
        if not os.path.isdir(filename):
            os.mkdir(filename)

        if episode is not None:
            torch.save(self.q_func.state_dict(), os.path.join(self.args.save_dir, "mymodel_" + str((episode // self.args.save_interval) % 10) + ".pth"))
        else:
            torch.save(self.q_func.state_dict(), os.path.join(self.args.save_dir, "bestmodel.pth"))

    def load_params(self):
        filename = os.path.join(self.args.save_dir, self.args.experiment, "mymodel.pth")
        if not os.path.isfile(filename):
            filename = os.path.join(self.args.save_dir, self.args.experiment, "bestmodel.pth")
        if torch.cuda.is_available():
            self.q_func.load_state_dict(torch.load(filename))
            self.q_func.to(torch.device("cuda"))
        else:
            self.q_func.load_state_dict(torch.load(filename, map_location='cpu'))


class Logger:

    def __init__(self, args):
        self.args = args
        self.loss_list = []
        self.reward_list = []
        self.reward_validation_list = []
        self.q_list = []

    def add_loss(self, loss):
        self.loss_list.append(loss)

    def add_reward(self, reward, is_valid=False):
        if is_valid:
            self.reward_validation_list.append(reward)
        else:
            self.reward_list.append(reward)

    def add_q(self, q):
        self.q_list.append(q)

    def plot_loss(self):
        plt.figure('loss')
        plt.plot(self.loss_list)
        plt.savefig(os.path.join(self.args.save_dir, 'loss.png'))

    def plot_reward(self):
        plt.figure('reward')
        plt.plot(self.reward_list)
        plt.savefig(os.path.join(self.args.save_dir, 'reward.png'))

        plt.figure('validation reward')
        plt.plot(self.reward_validation_list)
        plt.savefig(os.path.join(self.args.save_dir, 'validation_reward.png'))

    def plot_q(self):
        plt.figure('q')
        plt.plot(self.q_list)
        plt.savefig(os.path.join(self.args.save_dir, 'q.png'))

    def log_all(self):
        with open(os.path.join(self.args.save_dir, 'q.pkl'), 'wb') as f:
            pickle.dump(self.q_list, f)
        with open(os.path.join(self.args.save_dir, 'validation_reward.pkl'), 'wb') as f:
            pickle.dump(self.reward_validation_list, f)
        with open(os.path.join(self.args.save_dir, 'reward.pkl'), 'wb') as f:
            pickle.dump(self.reward_list, f)
        with open(os.path.join(self.args.save_dir, 'loss.pkl'), 'wb') as f:
            pickle.dump(self.loss_list, f)


class ReplayBuffer:
    
    def __init__(self, args):
        self.args = args
        self.state_list = [None] * self.args.max_capacity
        self.action_list = [None] * self.args.max_capacity
        self.is_end_list = [None] * self.args.max_capacity
        self.reward_list = [None] * self.args.max_capacity
        self.count = 0

    def rel_index(self, index):
        return index % self.args.max_capacity

    def add(self, state, action, reward, is_end):
        index = self.rel_index(self.count)
        self.state_list[index] = state
        self.action_list[index] = action
        self.reward_list[index] = reward
        self.is_end_list[index] = is_end
        self.count += 1

    def get_size(self):
        return min(self.count, self.args.max_capacity)

    def sample(self, num_samples):
        indices = np.random.choice(self.get_size(), num_samples, replace=False)
        sampled_states = [self.state_list[i] for i in indices]
        sampled_actions = [self.action_list[i] for i in indices]
        sampled_next_state = [self.state_list[self.rel_index(i + 1)] if not self.is_end_list[i] else None for i in indices]
        sampled_rewards = [self.reward_list[i] for i in indices]
        sampled_is_end = [self.is_end_list[i] for i in indices]
        return [sampled_states, sampled_actions, sampled_next_state, sampled_rewards, sampled_is_end]


# def test_policy():
#     args = parse_args()
#     args.logger = Logger(args)
#     policy = Policy(args)
#     replay_buffer = ReplayBuffer(args)

#     for i in range(2):
#         # create some sample state, next piece
#         next_piece = random.randint(0, 6)
#         field = np.zeros((rows + 4, cols))
#         field = field > 0
#         is_end = False
#         rows_cleared = 0
#         state = [field, next_piece]

#         # select action
#         action = policy.take_action(state)

#         # store transition
#         replay_buffer.add(state, action, rows_cleared, is_end)


# def test_replay_buffer():
#     args = parse_args()
#     args.logger = Logger(args)
#     replay_buffer = ReplayBuffer(args)

#     for game in range(100):
#         for t in range(50):
#             field = np.zeros((rows, cols))
#             state = [field, 0]
#             action = [0, 0]
#             rows_cleared = 0
#             is_end = True if t == 49 else False
#             replay_buffer.add(state, action, rows_cleared, is_end)


def train():

    args = parse_args()
    args.logger = Logger(args)
    policy = Policy(args)
    replay_buffer = ReplayBuffer(args)
    env = JengaEnv(args)

    for episode in range(args.num_episodes):

        if episode < args.learning_start:
            strategy = "random"
        else:
            if episode % args.save_interval == 0:
                strategy = "validation"
            else:
                strategy = "epsilon_greedy"

        # collect data
        reward_accum_list = []
        max_avg_reward = 0
        past_validation_steps_list = deque([])

        for game in range(args.num_games):


            env.reset()
            reward_accum = 0
            epsilon_greedy_start = 0  # initialize to 0 : do exploration first

            if args.sample_t and strategy == "epsilon_greedy" and len(past_validation_steps_list) == 10 and game >= 4:
                if game <= 7:
                    epsilon_greedy_start = np.random.choice(int(np.median(np.array(past_validation_steps_list)) / 2))
                else:
                    epsilon_greedy_start = np.random.choice(int(np.max(np.array(past_validation_steps_list)) / 2))

            
            for t in count():
                
                state = env.state

                if (strategy == "epsilon_greedy" and t < epsilon_greedy_start) or (strategy == "validation"):
                    action = policy.take_action(state, "validation")
                    print(action)
                else:
                    action = policy.take_action(state, strategy)
                    


                next_state, is_end = env.step(action)
                reward = env.get_reward(next_state, is_end)
                reward_accum += reward

                if strategy == "random" or (strategy == "epsilon_greedy" and t >= epsilon_greedy_start):
                    replay_buffer.add(state, action, reward, is_end)

                if is_end > 0:
                    print_str = "=" * 20 + " Episode " + str(episode) + "\tGame " + str(game) + " " + strategy + " " + "=" * 20 + " curr_height/max_height: " \
                                + str(next_state) + "/" + str(next_state.shape[0]) + "\tsample_t/total_t: " + str(epsilon_greedy_start) + "/" + str(t)
                    print(colored(print_str, 'red'))

                    reward_accum_list.append(reward_accum)

                    if strategy == "validation":
                        past_validation_steps_list.append(t)
                        if len(past_validation_steps_list) > 10:
                            past_validation_steps_list.popleft()
                    break

            if strategy == 'validation':
                break

                
                
                

        if strategy == "validation":
            policy.save_params(episode)

        if strategy != "random":
            args.logger.add_reward(np.average(reward_accum_list), is_valid=(strategy == "validation"))

        # learn
        if (episode >= args.learning_start) and (replay_buffer.get_size() > args.batch_size) and (strategy == "epsilon_greedy"):
            print("learning...")
            loss_list = []
            for i in range(args.num_games):
                samples = replay_buffer.sample(args.batch_size)
                loss = policy.learn(samples)
                loss_list.append(loss)
            args.logger.add_loss(np.average(loss_list))

        # make plot
        if strategy != "random":
            if len(args.logger.loss_list) % 100 == 0:
                args.logger.plot_loss()
                args.logger.plot_reward()
                args.logger.plot_q()
                args.logger.log_all()

        # update target q
        if (strategy != "random") and (episode % args.num_target_update_iter == 0):
            print("updating target q")
            policy.update_target_q()


if __name__ == "__main__":
    train()
    #args = parse_args()
    #test_env(args)