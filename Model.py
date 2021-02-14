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

'''m = Net(56,4,5)
X = torch.randn((2,5,56))
print(m(X,2).shape)'''
