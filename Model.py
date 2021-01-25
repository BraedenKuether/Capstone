import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.LSTM(NUM_FEATURES, 64, 1, batch_first = True)
        self.lin = nn.Linear(TIME_PERIOD_LENGTH * 64,NUM_ASSETS)
        self.soft_out = nn.Softmax(dim=1)
    def forward(self, x, batch_len):
        x, (h0, c0) = self.input(x)
        x = x.reshape((batch_len, TIME_PERIOD_LENGTH * 64))
        x = self.lin(x)
        x = self.soft_out(x)
        return x