import pyEX as p
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
class Portfolio:
    def __init__(self,assets,client):
        self.symbols = assets
        self.featurized = None 
        self.assetsByTime = []
        self.numAssets = len(assets)
        self.client = client 
        
        m = float('inf')
        tmp = []
        for a in assets:
            data = self.stockDF(self.client,a,0).sort_values(by = 'date')
            data["returns"] = data["close"].values-data["open"].values
            m = min(m,len(data))
            tmp.append(data)
        for a in tmp:
            self.assetsByTime.append(a[:m])
        
        #print("assets by time",self.assetsByTime)

        for a in self.assetsByTime:
            print(len(a)) 
        
        self.featurized,self.dates = self.featurize(self.assetsByTime)
         

    def printAssets(self):
        print(self.assetsByTime[0][:10,0])
        print(self.featurized[:10,0])

    def stockDF(self,client, symb, interval):
        #the time of data appears to be inconsistent
        #may need to check this down the road
        return client.chartDF(symb,timeframe='5y')
    
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
 

    def batch(self):
        pass             







