import torch
from torch.utils.data import Dataset

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class DailyDataset:

  def __init__(self,features,pctChange,bSize,numAssets,numFeatures,timePeriod,transformer=False,forecast=False):
    self.dailyReturns = pctChange
    X = features
    minx = torch.min(X,0).values
    maxx = torch.max(X,0).values
    X = (X-minx)/(maxx-minx)

    self.features = X 
    self.numAssets = numAssets
    self.numFeatures = numFeatures
    self.time = timePeriod
    self.transformer = transformer
    self.forecast = forecast
    self.bSize = bSize
    self.X,self.y = self.makeBatch(features) 
    

  def makeBatch(self,d):
    xs = []
    ys = []
    
    if self.forecast:
      size = len(self.features)-(self.time*2)
    else:
      size = len(self.features)-self.time

    for i in range(size):
      X = self.features[i:i+self.time] 
      if self.forecast:
        y = self.dailyReturns[i+self.time:i+self.time*2] 
      else:
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
    
