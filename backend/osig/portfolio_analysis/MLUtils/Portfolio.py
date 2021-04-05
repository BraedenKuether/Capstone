import pyEX as p
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from .Errors import *
class Portfolio:
  def __init__(self,assets,client,earnings = False):
    self.symbols = assets
    self.featurized = None 
    self.assetsByTime = []
    self.numAssets = len(assets)
    self.client = client
    #self.pctChange = None 
    if earnings:
      self.earnings_dfs = []
      earnings_feats = ['accountsPayable', 'commonStock', 'currentAssets', 'currentCash', 'currentLongTermDebt', 'inventory', 
                          'longTermInvestments', 'netTangibleAssets', 'otherAssets', 'otherCurrentAssets', 'otherCurrentLiabilities',
                          'otherLiabilities', 'propertyPlantEquipment', 'receivables', 'retainedEarnings', 'shareholderEquity',
                          'shortTermInvestments', 'totalAssets', 'totalCurrentLiabilities', 'totalLiabilities']
      self.num_earnings_features = len(earnings_feats) * self.numAssets
      for i,asset in enumerate(assets):

        try:
          df = client.balanceSheetDF(asset, period='quarter', last = 12).sort_index()[earnings_feats]
        except p.common.PyEXception as e:
          if "Response 404 - " in str(e):
            raise SymbolError
          elif "Response 402 - " in str(e):
            raise CreditError
          else:
            raise UknownError

        normalized_df = (df-df.mean())/df.std()
        normalized_df = normalized_df.fillna(0).replace(np.nan,0)
        self.earnings_dfs.append(normalized_df)
        if i == 0:
          min_date = self.earnings_dfs[i].index[0]
        min_date = max(min_date, self.earnings_dfs[i].index[0])
    
    m = float('inf')
    tmp = []
    for a in assets:

      try:
          data = self.stockDF(self.client,a).sort_values(by = 'date')
      except p.common.PyEXception as e:
          if "Response 404 - " in str(e):
            raise SymbolError
          elif "Response 402 - " in str(e):
            raise CreditError
          else:
            raise UknownError

      if earnings:
        data = data.loc[min_date:]
      data["returns"] = data["close"].values-data["open"].values
      
      m = min(m,len(data))
      tmp.append(data)

    for a in tmp:
      self.assetsByTime.append(a[:m])
    

    #for a in self.assetsByTime:
    #  print(len(a)) 
    
    self.featurized,self.dates = self.featurize(self.assetsByTime)

  def printAssets(self):
    print(self.assetsByTime[0][:10,0])
    print(self.featurized[:10,0])

  def stockDF(self,client, symb, timeframe='5y'):
    return client.chartDF(symb,timeframe=timeframe)
  
  def featurize(self,assets):
    feats = ['close','open','high','volume',\
         'uClose','uHigh','uLow','uVolume',\
         'fOpen','fClose','fHigh','fLow','fVolume']
    FV = []
    FD = []
    for a in assets:
      vals = []
      dates = []
      for f in feats:
          vals.append(list((a[f].values)))
      returns = list(a['close'].values-a['open'].values)
      vals.append(returns)
      dates.append(list(a.index.strftime("%Y-%m-%d").values))
      tens = torch.tensor(vals).T
      FV.append(tens)
      FD.append(np.transpose(dates))

    catted = torch.cat(FV,1)
    dates = np.concatenate(FD,1)
    return catted,dates
 
