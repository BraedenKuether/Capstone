import pyEX as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import json

import Portfolio as P
import Dataset 
import Model
from Trainer import *

IEX_TOKEN = "Tpk_647cd93d6c5842d6978e55c6f79b0e1a"
client = px.Client(IEX_TOKEN, version="sandbox")

class Tester:
  def __init__(self,P,timePeriod,batchSize,train_func = train_net):
    self.train_func = train_func
    self.portfolio = P
    self.time = timePeriod
    self.batch = batchSize
    self.numFeats = p.featurized.shape[1]
    print("train_func:", train_func)
    if self.train_func == train_net_earnings:
      self.dataset = Dataset.PortfolioDataSet(P.featurized,P.dates,timePeriod,P.numAssets,self.numFeats,batchSize,earnings=P.earnings_dfs,num_earning_feats=P.num_earnings_features)
    else:
      self.dataset = Dataset.PortfolioDataSet(P.featurized,P.dates,timePeriod,P.numAssets,self.numFeats,batchSize)
    self.trainModel()
    self.testingSet()


  def cumulativeReturns(self,weights,s=slice(None),withPlot=True):
    closes = [x['close'][s] for x in self.portfolio.assetsByTime]
    catted = pd.concat(closes,axis=1)
    returns = catted.pct_change()
    returns['pdr'] = returns.dot(weights)
    returns = (1+returns).cumprod() 
    if withPlot:
      returns['pdr'].plot(title="Cumulative Returns")
      plt.show()
    return returns
       

  def topbottom(self,weights):
    symbs = self.portfolio.symbols
    closes = [x['close'] for x in self.portfolio.assetsByTime]
    changes = map(lambda stock: (stock.iloc[-1]-stock.iloc[0])/stock.iloc[0], closes)
    changes = np.array(list(changes))
    withSyms = list(zip(changes,symbs))
    top = max(withSyms)
    bottom = min(withSyms)
    return (changes.dot(weights)) + 1,top,bottom 

  def performance(self,weights,timePeriod="ytd"):
    ytds = [client.chartDF(symbol=s,timeframe=timePeriod).sort_index()['close'] for s in self.portfolio.symbols]
    changes = map(lambda stock: (stock.iloc[-1]-stock.iloc[0])/stock.iloc[0], ytds)
    changes = np.array(list(changes))
    top = max(changes)
    top = min(changes)
    return ((changes.dot(weights)) + 1) 
        
  def spYTD(self,timePeriod="ytd"):
    spy = client.chartDF(symbol="spy",timeframe=timePeriod).sort_index()['close']
    return ((spy.iloc[-1]-spy.iloc[0])/spy.iloc[0]) + 1 

  def risk(self,weights):
    closes = [x['close'] for x in self.portfolio.assetsByTime]
    catted = pd.concat(closes,axis=1)
    dailyReturns = catted.pct_change()
            
    cov = (dailyReturns.cov())*252 # annulised by trading days

    var = np.dot(weights.T,np.dot(cov,weights))
            
    std = np.sqrt(var)

    return (var,std)

  def trainModel(self, epochs = 100):
    w, net,losses = self.train_func(self.dataset,self.portfolio.pctChange,self.time,self.portfolio.numAssets,self.numFeats,self.batch,epochs)
    self.net = net
    self.losses = losses
    self.epochs = epochs
    
  def plotLosses(self):
    #plt.plot(x,self.losses[i*self.epochs:(i+1)*self.epochs])
    plt.plot(self.losses)
    plt.show()
          
  def plotValidationLosses(self, show_last_x = -1):
    batches = int(len(self.valid_losses)/self.epochs)
    if show_last_x != -1:
      y = self.valid_losses[-1*show_last_x:]
    else:
      y = self.valid_losses
    x = self.valid_dates[-1*len(y):]
    plt.plot(x,y)
    ticks = [x[i] for i in range(0, len(x), 4)]
    plt.xticks(ticks, rotation=45)
    plt.show()
        
  
  def testingSet(self):
    #w,_ = train_net(self.dataset[:split],self.time,self.portfolio.numAssets,self.numFeats,self.batch)
    #w,net = train_net(self.dataset,self.time,self.portfolio.numAssets,self.numFeats,self.batch)
    #self.net = net
    #w = w.cpu()
    #self.cumulativeReturns(w,slice(split,None))
    if self.train_func == train_net_earnings:
      x,y,losses,dates = validation_set_earnings(self.dataset.testing_set,self.net,self.portfolio.numAssets,self.time)
      self.valid_losses = losses
      self.valid_dates = dates
    else:
      x,y = validation_set(self.dataset.testing_set,self.net,self.portfolio.numAssets,self.time)
    self.validation_returns(x,y)

  def cumulativeReturns(self,weights,s=slice(None),withPlot=True):
    closes = [x['close'][s] for x in self.portfolio.assetsByTime]
    catted = pd.concat(closes,axis=1)
    returns = catted.pct_change()
    returns['pdr'] = returns.dot(weights)
    returns = (1+returns).cumprod() 
    if withPlot:
      returns['pdr'].plot(title="Cumulative Returns")
      plt.show()
    return returns
      
  def validation_returns(self,x,y):
    plt.plot(x,y)
    plt.xticks(x, rotation=45)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    plt.show()
          
    
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
        
  def peRatio(self,withPlot=True):
    syms = self.portfolio.symbols 
    pes = []
    found = []
    for i,s in enumerate(syms):
      stats = client.advancedStatsDF(s)
      peR = stats['peRatio'][0]
      if peR != None:
        pes.append(peR)
        found.append(syms[i])
      if found and withPlot:
        plt.bar(range(len(pes)),pes)
        plt.xticks(range(len(found)),found)
        plt.show()
    return np.mean(pes),pes 


  def dividendYield(self,withPlot=True):
    syms = self.portfolio.symbols 
    divs = []
    found = []
    for i,s in enumerate(syms):
      stats = client.keyStatsDF(s)
      divY = stats['dividendYield'][0]
      if divY != None:
        divs.append(divY)
        found.append(syms[i])
    
    if found and withPlot:
      plt.bar(range(len(divs)),divs)
      plt.xticks(range(len(found)),found)
      plt.show()
    return syms 

  def psRatio(self,withPlot=True):
    syms = self.portfolio.symbols 
    ps = []
    found = []
    for i,s in enumerate(syms):
      stats = client.advancedStatsDF(s)
      psR = stats['priceToSales'][0]
      if psR != None:
        ps.append(psR)
        found.append(syms[i])

    if found and withPlot:
      plt.bar(range(len(ps)),ps)
      plt.xticks(range(len(found)),found)
      plt.show()
    return np.mean(ps),ps 


  def plotPortfolio(self,key="close"):
    plot = plt.gca()

    for asset in self.portfolio.assetsByTime:
      asset.plot(y=key,ax=plot)            

    plt.legend(self.portfolio.symbols)
    plot.set_title("Daily "+key)
    plt.show()


'''    
stonks = ['aapl', 'msft', 'fb', 'goog']
p = P.Portfolio(stonks,client)

ts = Tester(p,5,2)

print(ts.topbottom([.25,.25,.25,.25]))
'''
#stonks = ['vti', 'agg', 'dbc', 'vixy']
stonks = ['amd','wfc','ge','aapl','aal','hog','f','bac','t','intc']

'''
p = P.Portfolio(stonks,client)
ts = Tester(p,5,2)
ts.plotPortfolio()
ts.cumulativeReturns([1.0/len(stonks)]*len(stonks))
'''

p = P.Portfolio(stonks,client,earnings=True)
ts = Tester(p,30,20,train_func = train_net_earnings)
ts.plotPortfolio()
ts.plotLosses()
ts.plotValidationLosses()
'''
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
'''
