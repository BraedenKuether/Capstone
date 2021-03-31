import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import Portfolio as p
import pyEX as px
import matplotlib.pyplot as plt
from Trainer import sharpe_loss
from torch import optim

client = px.Client(version="sandbox")


bptt = 5
bsz = 2
def getBatch(source,i):
# days X assets*time
  ds = []
  cs = []
  minds = 100000
  mincs = 100000
  for j in range(i,i+bsz):
    seqLen = min(bptt,len(source)-1-j)
    data = source[j:j+seqLen,0::14]
    data = data.reshape(data.shape[0],1,data.shape[1])
    periodCloses = source[j+seqLen,0::14]
    minds = min(minds,data.shape[0])
    mincs = min(mincs,periodCloses.shape[0])
    cs.append(periodCloses)
    ds.append(data) 

  cleands = []
  cleancs = []
  for d,c in zip(ds,cs):
    if d.shape[0] == minds:
      cleands.append(d)
    if c.shape[0] == mincs:
      cleancs.append(c)
  batch = torch.cat(cleands,1)
  return batch,torch.stack(cleancs)

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)
  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

class TransformerModel(nn.Module):
  def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
    super(TransformerModel, self).__init__()
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    self.model_type = 'Transformer'
    self.pos_encoder = PositionalEncoding(ninp,dropout)
    encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
    self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
    self.ninp = ninp
    self.decoder = nn.Linear(ninp*bptt, ntoken)
    self.sm = nn.Softmax(dim=1)
    self.init_weights()

  def generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

  def init_weights(self):
    initrange = 0.1
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-initrange, initrange)

  def forward(self, src, mask):
    src = self.pos_encoder(src)
    #print(mask.shape)
    output = self.transformer_encoder(src,mask)
    #print(output.shape)
    output = output.reshape((output.shape[1],output.shape[0]*self.ninp))
    output = self.decoder(output)
    #print(output.shape)
    return output

def normalize(X):
  print(X[:10,::14])
  mins = torch.min(X,0).values
  maxs = torch.max(X,0).values
  print(mins.shape)
  mins = mins.reshape(1,-1)
  maxs = maxs.reshape(1,-1)
  normed = (X-mins)/(maxs-mins)
  print(mins)
  print(maxs)
  print(normed[:10,::14])
  return normed
          
    
stonks = ['aapl', 'msft', 'fb', 'goog']
batch = 2
port = p.Portfolio(stonks,client)
port.featurized = normalize(port.featurized)
print([x['close'][:11] for x in port.assetsByTime])
d,t = batches = getBatch(port.featurized,0)
print(d.shape,t.shape)
print(d)
print(t)
nAssets = 4
nFeats = 4 
nHead = 2 
nHid = 200
nLayers = 2
dropout = 0.2
model = TransformerModel(nAssets,nFeats,nHead,nHid,nLayers,dropout)
model = model.double()

crit = torch.nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0)
epochs = 100 

'''
X = torch.rand((5,2,1))
m = torch.rand((5,5))
o = torch.rand((2,4))
p = model(X,m)
loss = crit(p,o)
'''
last = [[],[]]
def train():
  model.train()
  mask = model.generate_square_subsequent_mask(bptt)
  losses = []
  for i in range(0,len(port.featurized)-bptt,bptt+1):
    data,closes= getBatch(port.featurized,i)
    for e in range(epochs):
      optimizer.zero_grad()
      if data.size(0) != bptt:
        mask = model.generate_square_subsequent_mask(data.size(0))
      out = model(data.double(),mask)
      #print(out,closes)
      loss = crit(out,closes)
      #print(loss)
      #torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
      #print(i)
      if i==1248:
        last[0] = out
        last[1] = closes 
        losses.append(loss.item())
      loss.backward()
      optimizer.step()
    plt.plot(losses)
    plt.show()

#train()
