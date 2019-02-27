import torch.nn.functional as F
from torch import nn

from settings import USE_FMINST


class NNetwork(nn.Module):

    def __init__(self):
        super(NNetwork, self).__init__()
        output = 10 if USE_FMINST else 40
        if not USE_FMINST:
            self.fc1 = nn.Linear(16384, 1024)
            self.fc2 = nn.Linear(1024, 784)
        self.fc3 = nn.Linear(784, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, output)

        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        if not USE_FMINST:
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = F.log_softmax(self.fc6(x), dim=1)
        return x


class CNNetwork(nn.Module):

    def __init__(self):
        super(CNNetwork, self).__init__()
        output = 10 if USE_FMINST else 40
        # 128x128
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # 64x64
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 32x32
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        multiplier = 7 if USE_FMINST else 32
        if USE_FMINST:
            self.fc1 = nn.Linear(32 * multiplier * multiplier, 564)
            self.fc2 = nn.Linear(564, output)
        else:
            self.fc1 = nn.Linear(16384, 1024)
            self.fc2 = nn.Linear(1024, output)
        # self.fc3 = nn.Linear(1024, 784)
        # self.fc4 = nn.Linear(784, 256)
        # self.fc5 = nn.Linear(256, 64)
        # self.fc6 = nn.Linear(64, output)

        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if not USE_FMINST:
            x = self.pool(F.relu(self.conv3(x)))
        multiplier = 3 if USE_FMINST else 32
        x = x.view(x.shape[0], -1)
        # x = x.view(-1, 64 * multiplier * multiplier)
        x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        # x = self.dropout(F.relu(self.fc3(x)))
        # x = self.dropout(F.relu(self.fc4(x)))
        # x = self.dropout(F.relu(self.fc5(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
