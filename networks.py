import torch.nn.functional as F
from torch import nn

from settings import USE_FMINST


class NNetwork(nn.Module):

    def __init__(self):
        super(NNetwork, self).__init__()
        output = 10 if USE_FMINST else 40

        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x
