import pyEX as px
from datetime import datetime
from Errors import *
import pandas as pd
from Grapher import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import json
import sys

import Portfolio as P
import Dataset 
from DailyDataset import DailyDataset
from Model import *
from Trainer import *

IEX_TOKEN = "Tpk_647cd93d6c5842d6978e55c6f79b0e1a"
client = px.Client(IEX_TOKEN, version="sandbox")


#print(px.account.metadataDF(token="pk_239736f61af84f61af18f0c68c920b06"))
#px.account.payAsYouGo(False)
#print(px.account.metadataDF())


class Tester:
  def __init__(self,P,timePeriod,batchSize,train_func = train_net, test_length = 126,epochs=100):
    self.train_func = train_func
    self.portfolio = P
    self.time = timePeriod
    self.batch = batchSize
    self.numFeats = self.portfolio.featurized.shape[1]
    print("train_func:", train_func)
    if self.train_func == train_net_earnings:
      self.dataset = DailyDataset(P.featurized,P.assetsByTime,self.batch,P.numAssets,self.numFeats,timePeriod,forecast=True,earnings=P.earnings_dfs,num_earning_feats=P.num_earnings_features,dates = P.dates)
      self.net = NetWithEarnings(self.numFeats,P.num_earnings_features,P.numAssets,timePeriod) 
    else:
      self.dataset = DailyDataset(P.featurized,P.assetsByTime,self.batch,P.numAssets,self.numFeats,timePeriod,forecast=True,dates = P.dates)
      self.net = Net(self.numFeats,P.numAssets,timePeriod)
    
    #self.trainModel(epochs=epochs)

  def cumulativeReturns(self,weights,s=slice(None),withPlot=True):
    closes = [x['close'][s] for x in self.portfolio.assetsByTime]
    catted = pd.concat(closes,axis=1)
    returns = catted.pct_change()
    returns['pdr'] = returns.dot(weights)
    returns = (1+returns).cumprod() 
    if withPlot:
      returns['pdr'].plot(title="Cumulative Returns")
      plt.show()
    
    returns = returns.dropna(0,'any')
    xs = returns['pdr'].index.strftime("%Y-%m-%d").values
    ys = returns['pdr'].values
    points = zip(xs,ys)

    dataDict = {"Cumulative Returns":points} 

    return toLine(dataDict) 
       
  
  def find(self,a,year):
    for i,x in enumerate(a):
      if year == x:
        return i

  def topbottom(self,weights):
    symbs = self.portfolio.symbols
    closes = [x['close'] for x in self.portfolio.assetsByTime]
    changes = map(lambda stock: (stock.iloc[-1]-stock.iloc[0])/stock.iloc[0], closes)
    changes = np.array(list(changes))
    withSyms = list(zip(changes,symbs))
    dataDict = {}
    for x,name in withSyms:
      dataDict[name] = x

    return toBar(dataDict)  

  def totalPerformance(self,weights):
    currentYear = datetime.today().year
    ytds = [s['close'] for s in self.portfolio.assetsByTime]
    changes = map(lambda stock: (stock.iloc[-1]-stock.iloc[0])/stock.iloc[0], ytds)
    changes = np.array(list(changes))
    changes = (changes.dot(weights)) + 1 
    return changes 
   
  def ytdPerformance(self,weights):
    currentYear = datetime.today().year
    ytds = [s['close'] for s in self.portfolio.assetsByTime]
    a = ytds[0].index.year
    start = self.find(a,currentYear)
    changes = map(lambda stock: (stock.iloc[-1]-stock.iloc[start])/stock.iloc[start], ytds)
    changes = np.array(list(changes))
    changes = (changes.dot(weights)) + 1 
    return changes 
        
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
    
    if found:
      dataDict = {}
      withSymbs = zip(pes,syms)
      for x,name in withSymbs:
        dataDict[name] = float(x)
      return toBar(dataDict),np.mean(pes) 


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
    
    if found:
      dataDict = {}
      withSymbs = zip(divs,syms)
      for x,name in withSymbs:
        dataDict[name] = float(x)
      return toBar(dataDict) 

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

    if found:
      dataDict = {}
      withSymbs = zip(ps,syms)
      for x,name in withSymbs:
        dataDict[name] = float(x)
      return toBar(dataDict),np.mean(ps) 


  def plotPortfolio(self,key="close"):
    plot = plt.gca()
    
    dataDict = {}
    for i,asset in enumerate(self.portfolio.assetsByTime):
      xs = asset[key].index.strftime("%Y-%m-%d").values
      ys = asset[key].values
      points = zip(xs,ys)
      dataDict[self.portfolio.symbols[i]] = points            

    plt.legend(self.portfolio.symbols)
    plot.set_title("Daily "+key)
    plt.show()
    return toLine(dataDict)

  def trainModel(self, epochs = 50):
    train,val,test = self.dataset.split(.8,.9)
    w, self.net, losses = self.train_func(self.net,train,epochs)
    
    self.losses = losses
    self.epochs = epochs
    if self.train_func == train_net_earnings:
      x,y,losses,dates = validation_set_earnings(test,self.net,self.portfolio.numAssets,self.time)
      self.valid_losses = losses
      self.valid_dates = dates
    else:
      x,y,losses,dates = validation_set(test,self.net,self.portfolio.numAssets,self.time)
      self.valid_losses = losses
      self.valid_dates = dates
      
    self.validation_returns(x,y)

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
      x,y,losses,dates = validation_set(self.dataset.testing_set,self.net,self.portfolio.numAssets,self.time)
      self.valid_losses = losses
      self.valid_dates = dates      
    self.validation_returns(x,y)

     
  def validation_returns(self,x,y):
    plt.plot(x,y)
    plt.xticks(x, rotation=45)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    plt.show()
          
    
  

'''    
stonks = ['aapl', 'msft', 'fb', 'goog']
p = P.Portfolio(stonks,client)

ts = Tester(p,5,2)

print(ts.topbottom([.25,.25,.25,.25]))
'''
stonks = ['vti', 'agg', 'dbc', 'vixy']
#stonks = ['amd','wfc','ge','aapl','aal','hog','f','bac','t','intc']
#stonks = ['adbe', 'atvi', 'axon', 'blk', 'bx', 'cost', 'crm', 'csco', 'cvs', 'dis', 'dpz', 'googl', 'hd', 'hon', 'jnj', 'jpm', 'lmt', 'mdt', 'nee', 'pxd', 'pypl', 'sbux', 'stz', 'swks', 't', 'twtr', 'usb', 'zts']
#stonks = ['adbe', 'atvi', 'blk', 'cost', 'crm', 'csco', 'cvs', 'dis', 'dpz', 'googl', 'hd', 'hon', 'jnj', 'jpm', 'lmt', 'mdt', 'nee', 'pxd', 'pypl', 'sbux', 'stz', 'swks', 't', 'twtr', 'usb', 'zts']

'''
p = P.Portfolio(stonks,client)
ts = Tester(p,5,2)
ts.plotPortfolio()
ts.cumulativeReturns([1.0/len(stonks)]*len(stonks))
'''

try:
  p = P.Portfolio(stonks,client,earnings=False)
except SymbolError:
  f = open("error.txt","w")
  f.write("{error : \"symbolNotFound\"}")
  sys.exit(-1)

p = P.Portfolio(stonks,client,earnings=False)
ts = Tester(p,10,60,train_func = train_net)

performancejson = ts.ytdPerformance([.25,.25,.25,.25])
topbjson = ts.topbottom([.25,.25,.25,.25])
perjson = ts.peRatio()
dyjson = ts.dividendYield()
psjson = ts.psRatio()
print(performancejson)
print(topbjson)
print(perjson)
print(dyjson)
print(psjson)
#f = open("graph.json","w")
#f.write(json)



'''
ts.plotPortfolio()
ts.plotLosses()
ts.plotValidationLosses()
'''
