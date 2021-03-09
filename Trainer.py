from Model import *
import torch
from torch import optim
import math
import time
import Dataset 
import Portfolio as p 
import pyEX as px
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

with open('token.json', 'r') as file:
    token = json.loads(file.read())['sandbox']
client = px.Client(token,version="sandbox")


torch.set_default_tensor_type('torch.FloatTensor')
def sharpe_loss(weights, returns):
  # weights batch * time * assets
  # returns batch * time * assets
  #print(weights.shape, returns.shape)

  #row wise dot product
  R = torch.sum(weights*returns,dim=-1)
  ER = torch.mean(R,1)
  STD = torch.std(R,1)
  ratio = torch.sum(ER/(STD+1e-6)) 
  return -ratio
  
  
def train_net(net,batches,epochs):
  #net = Net(numFeatures,numAssets,timePeriod).to('cuda')
  net = net.double().to('cuda')
  optimizer = optim.Adam(net.parameters(), lr=1e-6, weight_decay = 0)
  loss_fn = sharpe_loss
  lossVs = [] 
  weights = []
  for e in range(epochs):
    acc = 0.0
    if e % 10 == 0:
      print(e)
    for X,y,_,_ in batches:
      X = X.to('cuda')
      y = y.to('cuda')
      optimizer.zero_grad()
      out = net.forward(X)
      loss = loss_fn(out,y)     
      acc += loss.item()
      loss.backward()
      optimizer.step()
    lossVs.append(acc/len(batches))
  return weights,net,lossVs
  
#def train_net_earnings(d,returns,timePeriod,numAssets,numFeatures,batchSize,epochs):
def train_net_earnings(net,batches,epochs):
  #print(d)
  overall_val = 1
  start_day = 0
  #net = NetWithEarnings(numFeatures,d.NUM_EARNINGS_FEATURES,numAssets,timePeriod).to('cuda')
  net = net.double().to('cuda')
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
      X = X.double().to('cuda')
      y = y.double().to('cuda')
      earnings = earnings.double().to('cuda')
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
      X = X.double().to('cuda')
      y = y.double().to('cuda')
      out = net.forward(X)
      if simulation_day == 0:
        #print("input:", testing_d[i])
        #print(out)
        returns = y[-1][-1]
        weights = out[-1][-1].view(NUM_ASSETS)
        print("weights:", weights)
        #print("current:", testing_d.current_day_prices[i][-1][-1], "future:", testing_d.future_day_prices[i][-1][-1], "calculated change:", returns)
        print("calculated change:", returns, "day:", dates[-1][-1][0])
        percent_change = torch.dot(returns, weights)
        overall_val *= (1 + percent_change)
        print(overall_val)
        x.append(future_date[-1][-1][0])
        y_graph.append(overall_val.item())
      losses.append(loss_fn(out, y))
      losses_dates.append(dates[-1][-1][0])
      
    simulation_day += len(X)
    if simulation_day >= TIME_PERIOD_LENGTH:
      simulation_day = 0
      
  return x,y_graph,losses,losses_dates
  
def validation_set_earnings(batches,net,NUM_ASSETS,TIME_PERIOD_LENGTH):
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
      X = X.double().to('cuda')
      y = y.double().to('cuda')
      earnings = earnings.double().to('cuda')
      out = net.forward(X, earnings)
      if simulation_day == 0:
        returns = y[-1][-1]
        weights = out[-1][-1].view(NUM_ASSETS)
        print("weights:", weights)
        #print("current:", testing_d.current_day_prices[i][-1][-1], "future:", testing_d.future_day_prices[i][-1][-1], "calculated change:", returns, "day:", testing_d.dates[i][-1][-1][0])
        print("calculated change:", returns, "day:", dates[-1][-1][0])
        percent_change = torch.dot(returns, weights)
        overall_val *= (1 + percent_change)
        print(overall_val)
        x.append(future_date[-1][-1][0])
        y_graph.append(overall_val.item())
      losses.append(loss_fn(out, y))
      losses_dates.append(dates[-1][-1][0])
      
    simulation_day += len(X)
    if simulation_day >= TIME_PERIOD_LENGTH:
      simulation_day = 0
  return x,y_graph,losses,losses_dates

