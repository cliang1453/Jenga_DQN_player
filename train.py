import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import random

from environment import *
import torch
import torch.optim as optim
from torch.autograd import Variable


def parse_args():
    parser = argparse.ArgumentParser("Jenga DQN player")
    parser.add_argument("--init_height", type=int, default=10, help="the initial height of tower")
    args = parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
    test(args)