import torch.nn.functional as F
from torch import nn

from settings import USE_FMINST


class NNetwork(nn.Module):

    def __init__(self):
        super(NNetwork, self).__init__()
        output = 10 if USE_FMINST else 40

        # self.fc1 = nn.Linear(16384, 1024)
        # self.fc2 = nn.Linear(1024, 784)
        self.fc3 = nn.Linear(784, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, output)

        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = F.log_softmax(self.fc6(x), dim=1)
        return x
