import torch.nn as nn
import torch.nn.functional as F
import torch

class QFunc(nn.Module):
    def __init__(self, args):
        super(QFunc, self).__init__()

        if args.use_dir_info:
            self.conv1 = nn.Conv2d(2, 32, kernel_size=(2, 3), stride=1, padding=(0, 2))
        else:
            self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, 3), stride=1, padding=(0, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(2, 3), stride=1, padding=(0, 1))


        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.fc4 = nn.Linear((args.init_height*3-2-4)*3*128, (args.init_height*3-2)*6)
        self.fc5 = nn.Linear((args.init_height*3-2)*6, (args.init_height*3-2)*3)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)