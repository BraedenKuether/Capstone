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
def getBatch(normed,source,i):
  # days X assets*time
  ds = []
  rs = []
  for j in range(i,i+bsz):
    seqLen = min(bptt,len(source)-1-j)
    data = normed[j:j+seqLen]
    #print(data)
    data = data.reshape(data.shape[0],1,data.shape[1])
    #print(data)
    periodClose = source[j+(2*bptt-1),0::14]
    periodOpen = source[j+bptt,0::14]
    returns = (periodClose/periodOpen) - 1
    rs.append(returns)
    ds.append(data) 
  batch = torch.cat(ds,1)
  return batch,rs

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
    output = self.transformer_encoder(src,mask)
    output = output.reshape((output.shape[1],output.shape[0]*self.ninp))
    output = self.decoder(output)
    return self.sm(output)

def normalize(X):
  mins = torch.min(X,0).values
  maxs = torch.max(X,0).values
  mins = mins.reshape(1,-1)
  maxs = maxs.reshape(1,-1)
  normed = (X-mins)/(maxs-mins)
  return normed
   
stonks = ['aapl', 'msft', 'fb', 'goog']
batch = 2
port = p.Portfolio(stonks,client)
print([x['close'][:11] for x in port.assetsByTime])
source = port.featurized
port.featurized = normalize(port.featurized) 

nAssets = 4
nFeats = 56
nHead = 2
nHid = 200
nLayers = 2
dropout = 0.2
model = TransformerModel(nAssets,nFeats,nHead,nHid,nLayers,dropout)
model = model.double()

crit = sharpe_loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 50 

getBatch(port.featurized,source,1246)

X = torch.rand((5,2,56))
m = torch.rand((5,5))
#print(model(X,m).shape)
def train():
  model.train()
  mask = model.generate_square_subsequent_mask(bptt)
  losses = []
  for i in range(0,len(port.featurized)-(2*bptt-1)-1,bptt+1):
    data,rts = getBatch(port.featurized,source,i)
    if i % 100 == 0:
      print(i)
    for e in range(epochs):
      optimizer.zero_grad()
      if data.size(0) != bptt:
        mask = model.generate_square_subsequent_mask(data.size(0))
      out = model(data.double(),mask)
      #print(out,rts)
      loss = crit(out,0,out.size(0),rts,bptt)
      torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
      if e == epochs-1:
        losses.append(loss.item())
      loss.backward()
      optimizer.step()
  plt.plot(losses)
  plt.show()

train()
