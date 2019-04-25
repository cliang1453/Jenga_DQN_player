import torch.nn as nn
import torch.nn.functional as F
import torch

NUM_FEATURES = 2

class QFunc(nn.Module):
    def __init__(self, args):
        super(QFunc, self).__init__()

        if args.use_dir_info:
            self.conv1 = nn.Conv2d(2, 64, kernel_size=(2, 3), stride=1, padding=(0, 2))
        else:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 3), stride=1, padding=(0, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(2, 3), stride=1, padding=(0, 1))

        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, stride=1)
        self.fc4 = nn.Linear((args.init_height*3-2-4)*3*512, (args.init_height*3-2)*64)
        self.fc5 = nn.Linear((args.init_height*3-2)*64, (args.init_height*3-2)*3)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class QFunc_Heuristics(nn.Module):
    def __init__(self, args):
        super(QFunc_Heuristics, self).__init__()

        if args.use_dir_info:
            self.conv1 = nn.Conv2d(2, 64, kernel_size=(2, 3), stride=1, padding=(0, 2))
        else:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 3), stride=1, padding=(0, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(2, 3), stride=1, padding=(0, 1))

        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, stride=1)
        
        self.fc4 = nn.Linear((args.init_height*3-2-4)*3*512, (args.init_height*3-2)*64)
        self.fc_f = nn.Linear(NUM_FEATURES, (args.init_height*3-2)*64)

        self.fc5 = nn.Linear((args.init_height*3-2)*128, (args.init_height*3-2)*3)


    def forward(self, x, feature):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        y = F.relu(self.fc_f(feature))
        x = torch.cat((x, y), dim=1)

        return self.fc5(x)
        
