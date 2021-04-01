import torch
import pandas as pd
from torch.utils.data import Dataset
import datetime

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class DailyDataset:

  def __init__(self,features,assetsByTime,bSize,numAssets,numFeatures,timePeriod,transformer=False,forecast=False,earnings=None,num_earning_feats = None,dates=None):

    self.time = timePeriod
    self.transformer = transformer
    self.forecast = forecast
    self.NUM_EARNINGS_FEATURES = num_earning_feats
    self.dates = []
    self.future_dates = []

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
    self.X,self.y = self.makeBatch(features,earnings,dates)

 
  def getPctChange(self,assetsByTime):
    change = []
    if self.forecast:
      period = self.time
    else:
      period = 1

    for a in assetsByTime:
        x = a['close'].pct_change(periods=period).dropna().values
        #print(x[:10])
        x = torch.tensor(x).reshape(-1,1)
        change.append(x)
      
    return torch.cat(change,1)

  def split(self,train,val):
    trainL = int(len(self.X)*train) 
    valL = int(len(self.X)*val)
    
    if self.earnings:
      z = list(zip(self.X,self.y,self.earnings,self.dates,self.future_dates))
    else:
      z = list(zip(self.X,self.y,self.dates,self.future_dates))

    train = z[:trainL] 
    val = z[trainL:valL]
    test = z[trainL:]

    return train,val,test

  def makeBatch(self,d, earnings=None,dates=None):
    xs = []
    ys = []
    
    if earnings:
      for i in range(len(earnings)):
        earnings[i] = earnings[i].sort_index()
      raw_earnings = earnings.copy()
      self.earnings = []
    else:
      self.earnings = None
    
    size = len(self.features)-self.time

    for i in range(size):
      X = self.features[i:i+self.time] 
      y = self.dailyReturns[i:i+self.time] 

      X = torch.unsqueeze(X,0)
      y = torch.unsqueeze(y,0)
      
      xs.append(X)
      ys.append(y)
      if earnings:
        last_date = datetime.datetime.strptime(dates[i + self.time - 1][0], '%Y-%m-%d')
        earnings_list = []
        for earning in raw_earnings:
          latest_earning = earning.loc[:last_date].iloc[-1]
          earnings_list.extend(latest_earning.tolist())
        self.earnings.append(earnings_list)
      self.dates.append(dates[i:i+self.time].tolist())
      self.future_dates.append(dates[i+self.time:i + 2*self.time].tolist())

    X,y = torch.cat(xs,0),torch.cat(ys,0)

    X,y = torch.split(X,self.bSize),torch.split(y,self.bSize)

    #if size % self.bSize:
    #  X = X[:-1]
    #  y = y[:-1]
    if earnings:
      self.earnings=torch.split(torch.Tensor(self.earnings),self.bSize)
    self.dates = [self.dates[i:i+self.bSize] for i in range(0,len(self.dates),self.bSize)]
    self.future_dates = [self.future_dates[i:i+self.bSize] for i in range(0,len(self.future_dates),self.bSize)]
    
    return X,y

  def __len__(self):
    return len(self.X)
  
  def __getitem__(self,i):
    X,y,earnings = self.X[i],self.y[i].to('cuda'),self.earnings[i].to('cuda')

    if self.transformer:
      X = torch.transpose(X,0,1).to('cuda')
      
    if self.earnings:
      return X,y,earnings
    else:
      return X,y 
