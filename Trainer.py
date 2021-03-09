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
    for X,y in batches:
      optimizer.zero_grad()
      out = net.forward(X)
      loss = loss_fn(out,y)
      acc += loss.item()
      loss.backward()
      optimizer.step()
    lossVs.append(acc/len(batches))
  return weights,net,lossVs
  
def train_net_earnings(d,returns,timePeriod,numAssets,numFeatures,batchSize,epochs):
  #print(d)
  overall_val = 1
  start_day = 0
  net = NetWithEarnings(numFeatures,d.NUM_EARNINGS_FEATURES,numAssets,timePeriod).to('cuda')
  losses_new_net = []
  num_epochs = epochs
  optimizer = optim.Adam(net.parameters(), lr=1e-7, weight_decay = 0)
  loss_fn = sharpe_loss
  total_time = 0
  simulation_day = 0
  weights = []
  for epoch in range(num_epochs):
    start = time.time()
    epoch_loss = 0
    #("step {}".format(i))
    for i in range(len(d)):
      out = net.forward(d[i], d.earnings[i], len(d[i]))

      future_index = math.ceil(i + (timePeriod/batchSize))
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
      loss = loss_fn(out, d.future_returns(i))
      epoch_loss += loss.item()/len(d[i])
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    losses_new_net.append(epoch_loss/len(d))

    total_time += time.time() - start
    avg_time = total_time/(i + 1)
    #print("eta: {}m {}s".format(int(avg_time/60 * (len(d) - i - 1)), int((avg_time *(len(d) - i - 1)) % 60 )))
    simulation_day += len(d[i])
    if simulation_day >= timePeriod:
      simulation_day = 0

  #print(overall_val)
  return weights,net,lossVs


def validation_set(testing_d,net,NUM_ASSETS,TIME_PERIOD_LENGTH):
  loss_fn = sharpe_loss
  overall_val = 1
  simulation_day = 0
  x = [testing_d.dates[0][-1][-1][0]]
  y = [1]
  losses = []
  losses = []
  losses_dates = []
  losses_dates = []
  for i in range(len(testing_d)):
    with torch.no_grad():
      if simulation_day == 0:
        #print("input:", testing_d[i])
        out = net.forward(testing_d[i], len(testing_d[i]))
        #print(out)
        returns = testing_d.future_returns(i)[-1][-1]
        weights = out[-1][-1].view(NUM_ASSETS)
        print("weights:", weights)
        print("current:", testing_d.current_day_prices[i][-1][-1], "future:", testing_d.future_day_prices[i][-1][-1], "calculated change:", returns)
        percent_change = torch.dot(returns, weights)
        overall_val *= (1 + percent_change)
        print(overall_val)
        x.append(testing_d.future_dates[i][-1][-1][0])
        y.append(overall_val.item())
      losses.append(loss_fn(out, testing_d.future_returns(i)))
      losses_dates.append(testing_d.dates[i][-1][-1][0])
      
    simulation_day += len(testing_d[i])
    if simulation_day >= TIME_PERIOD_LENGTH:
      simulation_day = 0
      
  return x,y,losses,losses_dates
  
def validation_set_earnings(testing_d,net,NUM_ASSETS,TIME_PERIOD_LENGTH):
  loss_fn = sharpe_loss
  overall_val = 1
  simulation_day = 0
  x = [testing_d.dates[0][-1][-1][0]]
  y = [1]
  losses = []
  losses_dates = []
  print(testing_d.dates[0])
  for i in range(len(testing_d)):
    with torch.no_grad():
      if simulation_day == 0:
        #print("input:", testing_d[i])
        out = net.forward(testing_d[i], testing_d.earnings[i], len(testing_d[i]))
        #print(out)
        returns = testing_d.future_returns(i)[-1][-1]
        weights = out[-1][-1].view(NUM_ASSETS)
        print("weights:", weights)
        print("current:", testing_d.current_day_prices[i][-1][-1], "future:", testing_d.future_day_prices[i][-1][-1], "calculated change:", returns, "day:", testing_d.dates[i][-1][-1][0])
        percent_change = torch.dot(returns, weights)
        overall_val *= (1 + percent_change)
        print(overall_val)
        x.append(testing_d.future_dates[i][-1][-1][0])
        y.append(overall_val.item())
      losses.append(loss_fn(out, testing_d.future_returns(i)))
      losses_dates.append(testing_d.dates[i][-1][-1][0])
      
    simulation_day += len(testing_d[i])
    if simulation_day >= TIME_PERIOD_LENGTH:
      simulation_day = 0
  return x,y,losses,losses_dates

