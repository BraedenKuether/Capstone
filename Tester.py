import pyEX as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import Portfolio as p

class Tester:

    def __init__(self,P):
        self.portfolio = P

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


    def plotPortfolio(self,key="close"):
        plot = plt.gca()

        for asset in self.portfolio.assetsByTime:
            asset.plot(y=key,ax=plot)            

        legend(self.portfolio.symbols)
        plot.set_title("Daily "+key)
        plt.show()


client = px.Client(version="sandbox")
stonks = ['aapl', 'csco', 'ibm', 'amzn']

p = p.Portfolio(stonks,client)

ts = Tester(p)

x = ts.sharpe(np.array([.35,.15,.15,.35]))
print(x)
