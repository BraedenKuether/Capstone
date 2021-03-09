import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,NUM_FEATURES,NUM_ASSETS,TIME_PERIOD_LENGTH):
        super(Net, self).__init__()
        self.time = TIME_PERIOD_LENGTH
        self.input = nn.LSTM(NUM_FEATURES, 64, 1, batch_first = True)
        self.lin = nn.Linear(64,NUM_ASSETS)
        self.soft_out = nn.Softmax(dim=2)

    def forward(self, x):
        # x : batch_len X self.time X NUM_FEATURES
        x, (hn, cn) = self.input(x)
        # batch X time x assets*time
        #print(x.shape)
        x = self.lin(x)
        #print(x.shape)
        x = self.soft_out(x)
        #print(x.shape)
        #print(x)
        return x

class NetWithEarnings(nn.Module):
  def __init__(self,NUM_FEATURES,NUM_EARNING_FEATURES,NUM_ASSETS,TIME_PERIOD_LENGTH):
    super(NetWithEarnings, self).__init__()
    self.time = TIME_PERIOD_LENGTH
    self.assets = NUM_ASSETS
    self.input = nn.LSTM(NUM_FEATURES, 64, 1, batch_first = True)
    self.relu = nn.ReLU()
    self.lin = nn.Linear(64,NUM_ASSETS)
    self.earnings_lin1 = nn.Linear(NUM_EARNING_FEATURES, NUM_EARNING_FEATURES)
    self.earnings_lin2 = nn.Linear(NUM_EARNING_FEATURES, NUM_ASSETS*TIME_PERIOD_LENGTH)
    self.final_lin = nn.Linear(NUM_ASSETS*2, NUM_ASSETS)
    self.soft_out = nn.Softmax(dim=2)
  def forward(self, x, earnings):
    #print("input:",x,earnings)
    x, (h0, c0) = self.input(x)
    #x = x.reshape((batch_len, self.time* 64))
    x = self.relu(x)
    x = self.lin(x)
    x = self.relu(x)
    earnings = self.earnings_lin1(earnings)
    earnings = self.relu(earnings)
    earnings = self.earnings_lin2(earnings)
    earnings = self.relu(earnings)
    earnings = earnings.reshape((earnings.shape[0], self.time, self.assets))
    combined = torch.cat((x,earnings), dim = 2)
    #print(x.shape, combined.shape, self.assets)
    x = self.final_lin(combined)
    x = self.soft_out(x)
    return x

