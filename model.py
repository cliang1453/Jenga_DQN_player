import torch.nn as nn
import torch.nn.functional as F
import torch

class QFunc(nn.Module):
    def __init__(self, args):
        super(QFunc, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(64 * 28, 256)
        self.fc5 = nn.Linear(256, 1)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)

        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        #print(x.shape)
        return self.fc5(x)