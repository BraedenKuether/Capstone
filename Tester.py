import pyEX as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import json

import Portfolio as p
import Dataset 
import Model
from Trainer import sharpe_loss,train_net 

client = px.Client(version="sandbox")

class Tester:

    def __init__(self,P,timePeriod,batchSize):
        self.portfolio = P
        self.time = timePeriod
        self.batch = batchSize
        self.numFeats = p.featurized.shape[1]
        print(self.numFeats)
        self.dataset = Dataset.PortfolioDataSet(P.featurized,timePeriod,P.numAssets,self.numFeats,batchSize)

    def runModel(self):
        train_net(self.dataset,self.time,self.portfolio.numAssets,self.numFeats,self.batch)
        
    def cumulativeReturns(self,weights,withPlot=True):
        closes = [x['close'] for x in self.portfolio.assetsByTime]
        catted = pd.concat(closes,axis=1)
        returns = catted.pct_change()
        returns['pdr'] = returns.dot(weights)
        returns = (1+returns).cumprod() 
        if withPlot:
            returns['pdr'].plot(title="Cumulative Returns")
            plt.show()
        return returns
    
    def risk(self,weights):
        closes = [x['close'] for x in self.portfolio.assetsByTime]
        catted = pd.concat(closes,axis=1)
        dailyReturns = catted.pct_change()
        
        cov = (dailyReturns.cov())*252 # annulised by trading days
        
        var = np.dot(weights.T,np.dot(cov,weights))
        
        std = np.sqrt(var)

        return (var,std)
    
    def sharpe(self,weights,riskFree=0.0):
         
        returns = [x['close']/x.iloc[0]['close'] for x in self.portfolio.assetsByTime]  
        returns = pd.concat(returns,axis=1)

        returns['total'] = returns.dot(weights)
        returns['daily return'] = returns['total'].pct_change()

        mu = returns['daily return'].mean()
        std = returns['daily return'].std()
        sharpe = mu/std

        return (252**.5)*sharpe

    
    def alphabeta(self,weights,market="SPY",withPlot=True):
        closes = [x['close'] for x in self.portfolio.assetsByTime]
        closes = pd.concat(closes,axis=1)
        
        returns = closes.pct_change()[1:]
        returns = returns.dot(weights)

        benchMark = client.chartDF(market,timeframe='5y')['close'].pct_change()[1:]
        
        beta,alpha = stats.linregress(benchMark.values,returns.values)[0:2]
        
        if withPlot:
            x = benchMark.values
            y = returns.values
            plt.plot(x,y,'o',label="returns")
            plt.plot(x,alpha + beta*x, 'r', label="fitted line")
            plt.legend()
            plt.show()
        return (round(alpha,4),round(beta,4))
        

    def plotPortfolio(self,key="close"):
        plot = plt.gca()

        for asset in self.portfolio.assetsByTime:
            asset.plot(y=key,ax=plot)            

        legend(self.portfolio.symbols)
        plot.set_title("Daily "+key)
        plt.show()


stonks = ['vti', 'agg', 'dbc', 'vixy']
p = p.Portfolio(stonks,client)

ts = Tester(p,5,2)

r = ts.cumulativeReturns([.25,.25,.25,.25])
r = r.dropna(0,'any')
jsons = []
xs = r['pdr'].index.strftime("%Y-%m-%d").values
#ys = [x for x in r['pdr'].values if not np.isnan(x)]
ys = r['pdr'].values
print(ys)
points = zip(xs,ys)
d = {"id":"Cumulative Portfolio Returns",
     "color": "hsl(29,70%,50%)",
     "data":[{"x":x, "y":y} for x,y in points]}
jsons.append(d)

jstr = json.dumps(jsons)

f = open('graph.json',"w")
f.write(jstr)
f.close()
