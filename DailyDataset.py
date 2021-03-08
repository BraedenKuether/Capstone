import torch
import pandas as pd
from torch.utils.data import Dataset

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class DailyDataset:

  def __init__(self,features,assetsByTime,bSize,numAssets,numFeatures,timePeriod,transformer=False,forecast=False):

    self.time = timePeriod
    self.transformer = transformer
    self.forecast = forecast

    self.dailyReturns = self.getPctChange(assetsByTime) 
    X = features
    if self.forecast:
      X = X[:-timePeriod]

    minx = torch.min(X,0).values
    maxx = torch.max(X,0).values
    X = (X-minx)/(maxx-minx)

    self.features = X 
    self.numAssets = numAssets
    self.numFeatures = numFeatures
    self.bSize = bSize
    self.X,self.y = self.makeBatch(features) 

 
  def getPctChange(self,assetsByTime):
    change = []
    if self.forecast:
      period = self.time
    else:
      period = 1

    for a in assetsByTime:
        x = a['close'].pct_change(periods=period).dropna().values
        print(x[:10])
        x = torch.tensor(x).reshape(-1,1)
        change.append(x)
      
    return torch.cat(change,1)

  def split(self,train,val):
    trainL = int(len(self.X)*train) 
    valL = int(len(self.X)*val)
    
    z = list(zip(self.X,self.y))

    train = z[:trainL] 
    val = z[trainL:valL]
    test = z[valL:]

    return train,val,test

  def makeBatch(self,d):
    xs = []
    ys = []
    
    size = len(self.features)-self.time

    for i in range(size):
      X = self.features[i:i+self.time] 
      y = self.dailyReturns[i:i+self.time] 

      X = torch.unsqueeze(X,0)
      y = torch.unsqueeze(y,0)
      
      xs.append(X)
      ys.append(y)

    X,y = torch.cat(xs,0),torch.cat(ys,0)

    X,y = torch.split(X,self.bSize),torch.split(y,self.bSize)

    if size % self.bSize:
      X = X[:-1]
      y = y[:-1]
    
    return X,y

  def __len__(self):
    return len(self.X)
  
  def __getitem__(self,i):
    X,y = self.X[i],self.y[i]

    if self.transformer:
      X = torch.transpose(X,0,1)

    return X,y 
