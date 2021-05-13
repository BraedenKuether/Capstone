from .Model import *
import torch
from torch import optim
import math
import time
from . import Dataset 
from . import Portfolio as p 
import pyEX as px
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

#with open('token.json', 'r') as file:
#    token = json.loads(file.read())['sandbox']
#client = px.Client(token,version="sandbox")
IEX_TOKEN = "Tpk_647cd93d6c5842d6978e55c6f79b0e1a"
client = px.Client(IEX_TOKEN, version="sandbox")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch.set_default_tensor_type('torch.FloatTensor')
def sharpe_loss(weights, returns):
  '''
    computes the sharpe loss based on the paper
    https://jfds.pm-research.com/content/iijjfds/2/4/8.full.pdf
  # weights batch * time * assets
  # returns batch * time * assets
  '''

  #row wise dot product
  R = torch.sum(weights*returns,dim=-1)
  ER = torch.mean(R,1)
  STD = torch.std(R,1)
  ratio = torch.sum(ER/(STD+1e-6)) 
  return -ratio
  
  
def train_net(net,batches,epochs):
  '''
  trains net using gradient descent
  returns:
    weights: array of weights corresponding to each asset
    net: the model with updated weights from training
    lossVs: array of per epoch losses
  '''

  #net = Net(numFeatures,numAssets,timePeriod).to(DEVICE)
  net = net.double().to(DEVICE)
  optimizer = optim.Adam(net.parameters(), lr=1e-6, weight_decay = 0)
  loss_fn = sharpe_loss
  lossVs = [] 
  weights = []
  for e in range(epochs):
    acc = 0.0
    if e % 10 == 0:
      print(e)
    for X,y,_,_ in batches:
      X = X.to(DEVICE)
      y = y.to(DEVICE)
      optimizer.zero_grad()
      out = net.forward(X)
      loss = loss_fn(out,y)     
      acc += loss.item()
      loss.backward()
      optimizer.step()
    lossVs.append(acc/len(batches))
  return weights,net,lossVs
  
def train_net_earnings(net,batches,epochs):
  '''
    same idea but for the net earnings model
    returns:
      weights: array of weights corresponding to each asset
      net: the model with updated weights from training
      losses_new_net: array of per epoch losses
  '''
  overall_val = 1
  start_day = 0
  net = net.double().to(DEVICE)
  losses_new_net = []
  optimizer = optim.Adam(net.parameters(), lr=1e-6, weight_decay = 0)
  loss_fn = sharpe_loss
  total_time = 0
  simulation_day = 0
  weights = []
  for epoch in range(epochs):
    start = time.time()
    epoch_loss = 0
    #("step {}".format(i))
    for X,y,earnings,_,_ in batches:
      X = X.double().to(DEVICE)
      y = y.double().to(DEVICE)
      earnings = earnings.double().to(DEVICE)
      out = net.forward(X, earnings)

      '''
      if epoch == 0 and simulation_day == 0 and future_index < len(d):
        with torch.no_grad():
          future_index = math.ceil(i + (timePeriod/batchSize))
          sim_out = net.forward(d[future_index], d.earnings[future_index], len(d[future_index]))
          weights = sim_out[0].view(numAssets)
          percent_change = torch.dot(d.future_returns(future_index)[0], weights)
          overall_val *= 1 + percent_change
          print("return:",overall_val)
          #print("allocs: ",weights)
      '''
      #loss = loss_fn(out, i, len(d[i]), d.future_returns(i), timePeriod)
      loss = loss_fn(out, y)
      epoch_loss += loss.item()/len(X)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    losses_new_net.append(epoch_loss/len(batches))

    #total_time += time.time() - start
    #avg_time = total_time/(i + 1)
    #print("eta: {}m {}s".format(int(avg_time/60 * (len(d) - i - 1)), int((avg_time *(len(d) - i - 1)) % 60 )))

  #print(overall_val)
  return weights,net,losses_new_net


def validation_set(batches,net,NUM_ASSETS,TIME_PERIOD_LENGTH):
  '''
    computes the cumulative return of the model over a given time period
    
    params:
      NUM_ASSETS: # assets in the portfolio
      TIME_PERIOD_LENGTH: # days the returns are computed over
      batches: daily features,returns,and dates for each asset
      net: trained model
    
    returns:
      x: array of dates evaluted over
      y_graph: array of cumulative returns over the time period
      losses: array of model losses
      losses_dates: corresponding date labels for losses
      weights: weights used for model
  '''
  loss_fn = sharpe_loss
  overall_val = 1
  simulation_day = 0
  _,_,dates,_ = batches[0]
  x = [dates[-1][-1][0]]
  y_graph = [1]
  losses = []
  losses_dates = []
  for X,y,dates,future_date in batches:
    with torch.no_grad():
      X = X.double().to(DEVICE)
      y = y.double().to(DEVICE)
      out = net.forward(X)
      if simulation_day == 0:
        #print("input:", testing_d[i])
        #print(out)
        returns = y[-1][-1]
        weights = out[-1][-1].view(NUM_ASSETS)
        #print("weights:", weights)
        #print("current:", testing_d.current_day_prices[i][-1][-1], "future:", testing_d.future_day_prices[i][-1][-1], "calculated change:", returns)
        #print("calculated change:", returns, "day:", dates[-1][-1][0])
        percent_change = torch.dot(returns, weights)
        overall_val *= (1 + percent_change)
        #print(overall_val)
        x.append(future_date[-1][-1][0])
        y_graph.append(overall_val.item())
      losses.append(loss_fn(out, y))
      losses_dates.append(dates[-1][-1][0])
      
    simulation_day += len(X)
    if simulation_day >= TIME_PERIOD_LENGTH:
      simulation_day = 0
  
  print(len(batches[-1]))
  last_x, _, _, _ = batches[-1]
  last_x = last_x.double().to(DEVICE)
  weights = net.forward(last_x)[-1][-1].view(NUM_ASSETS)
  weights = weights.tolist()
  return x,y_graph,losses,losses_dates, weights
  
def validation_set_earnings(batches,net,NUM_ASSETS,TIME_PERIOD_LENGTH):
  '''
    same idea as above but is used with a earnings model 
    params:
      NUM_ASSETS: # assets in the portfolio
      TIME_PERIOD_LENGTH: # days the returns are computed over
      batches: daily features,returns,and dates for each asset
      net: trained model
    
    returns:
      x: array of dates evaluted over
      y_graph: array of cumulative returns over the time period
      losses: array of model losses
      losses_dates: corresponding date labels for losses
      weights: weights used for model
  '''
  loss_fn = sharpe_loss
  overall_val = 1
  simulation_day = 0
  _,_,_,dates,_ = batches[0]
  x = [dates[-1][-1][0]]
  #print(x)
  y_graph = [1]
  losses = []
  losses_dates = []
  for X,y,earnings,dates,future_date in batches:
    with torch.no_grad():
      X = X.double().to(DEVICE)
      y = y.double().to(DEVICE)
      earnings = earnings.double().to(DEVICE)
      out = net.forward(X, earnings)
      if simulation_day == 0:
        returns = y[-1][-1]
        weights = out[-1][-1].view(NUM_ASSETS)
        #print("weights:", weights)
        #print("current:", testing_d.current_day_prices[i][-1][-1], "future:", testing_d.future_day_prices[i][-1][-1], "calculated change:", returns, "day:", testing_d.dates[i][-1][-1][0])
        #print("calculated change:", returns, "day:", dates[-1][-1][0])
        percent_change = torch.dot(returns, weights)
        overall_val *= (1 + percent_change)
        #print(overall_val)
        x.append(future_date[-1][-1][0])
        y_graph.append(overall_val.item())
      losses.append(loss_fn(out, y))
      losses_dates.append(dates[-1][-1][0])
      
    simulation_day += len(X)
    if simulation_day >= TIME_PERIOD_LENGTH:
      simulation_day = 0
      
  print(len(batches[-1]))
  last_x, _, last_earnings, _, _ = batches[-1]
  last_x = last_x.double().to(DEVICE)
  last_earnings = last_earnings.double().to(DEVICE)
  weights = net.forward(last_x, last_earnings)[-1][-1].view(NUM_ASSETS)
  weights = weights.tolist()
  return x,y_graph,losses,losses_dates, weights

