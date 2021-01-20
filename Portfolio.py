import pyEX as p
import torch
import pandas as pd
import matplotlib.pyplot as plt
class Portfolio:
    def __init__(self,assets,client):
        self.symbols = assets
        self.featurized = None 
        self.assetsByTime = []
        self.numAssets = len(assets)
        self.client = client 

        for a in assets:
            self.assetsByTime.append(self.stockDF(self.client,a,0))

        self.featurized = self.featurize(self.assetsByTime)

    def printAssets(self):
        print(self.assetsByTime[0][:10,0])
        print(self.featurized[:10,0])

    def stockDF(self,client, symb, interval):
        #the time of data appears to be inconsistent
        #may need to check this down the road
        return client.chartDF(symb,timeframe='5y')
    
    def featurize(self,assets):
        #NEED TO ADD RETURNS
        feats = ['close','open','high','volume',\
             'uClose','uHigh','uLow','uVolume',\
             'fOpen','fClose','fHigh','fLow','fVolume']
        FV = []
        for a in assets:
            vals = []
            for f in feats:
                vals.append(list((a[f].values)))
            tens = torch.tensor(vals).T
            FV.append(tens)
        catted = torch.cat(FV,1)
        return catted 
 

    def batch(self):
        pass             


client = p.Client(version="sandbox")
stonks = ['vti', 'agg', 'dbc', 'vixy']

p = Portfolio(stonks,client)
#p.printAssets()





