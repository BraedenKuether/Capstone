import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,NUM_FEATURES,NUM_ASSETS,TIME_PERIOD_LENGTH):
        super(Net, self).__init__()
        self.time = TIME_PERIOD_LENGTH
        self.input = nn.LSTM(NUM_FEATURES, 64, 1, batch_first = True)
        self.lin = nn.Linear(self.time*64,NUM_ASSETS)
        self.soft_out = nn.Softmax(dim=1)

    def forward(self, x, batch_len):
        # x : batch_len X self.time X (NUM_ASSETS*TIME_PERIOD_LENGTH)
        x, (hn, cn) = self.input(x)
        x = x.reshape((batch_len, self.time* 64))
        x = self.lin(x)
        print(x.shape)
        x = self.soft_out(x)
        return x

class NetWithEarnings(nn.Module):
  def __init__(self,NUM_FEATURES,NUM_EARNING_FEATURES,NUM_ASSETS,TIME_PERIOD_LENGTH):
    super(NetWithEarnings, self).__init__()
    self.time = TIME_PERIOD_LENGTH
    self.input = nn.LSTM(NUM_FEATURES, 64, 1, batch_first = True)
    self.lin = nn.Linear(TIME_PERIOD_LENGTH * 64,NUM_ASSETS)
    self.earnings_lin1 = nn.Linear(NUM_EARNING_FEATURES, NUM_EARNING_FEATURES)
    self.earnings_lin2 = nn.Linear(NUM_EARNING_FEATURES, NUM_ASSETS)
    self.final_lin = nn.Linear(NUM_ASSETS*2, NUM_ASSETS)
    self.soft_out = nn.Softmax(dim=1)
  def forward(self, x, earnings, batch_len):
    #print("input:",x,earnings)
    x, (h0, c0) = self.input(x)
    x = x.reshape((batch_len, self.time* 64))
    x = self.lin(x)
    earnings = self.earnings_lin1(earnings)
    earnings = self.earnings_lin2(earnings)
    combined = torch.cat((x,earnings), dim = 1)
    x = self.final_lin(combined)
    x = self.soft_out(x)
    return x

