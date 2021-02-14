from Model import *
import torch
from torch import optim
import math
import time

torch.set_default_tensor_type('torch.cuda.FloatTensor')
def sharpe_loss(weights, batch_pos, batch_len, returns, TIME_PERIOD_LENGTH):
  total_ratio = 0
  er = 0
  er2 = 0
  annual_risk_free_rate = 1.09
  daily_risk_free_rate = annual_risk_free_rate ** (1/365)
  er_list = []
  for batch in range(batch_len):
    r_i_t = returns[batch]
    curr_er = torch.dot(r_i_t, weights[batch])
    er += (curr_er - daily_risk_free_rate**TIME_PERIOD_LENGTH)
    er2 += curr_er**2
    er_list.append(curr_er)
  er_list = torch.Tensor(er_list)
  er = er/batch_len
  er2 = er2/batch_len
  ratio = er / torch.std(er_list)
  ratio = -1 * ratio
  print(ratio)
  return ratio
  
  
def train_net(d,timePeriod,numAssets,numFeatures,batchSize):
  #print(d)
  overall_val = 1
  start_day = 0
  net = Net(numFeatures,numAssets,timePeriod).to('cuda')
  losses_new_net = []
  num_epochs = 100
  optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay = 0)
  loss_fn = sharpe_loss
  total_time = 0
  simulation_day = 0
  weights = []
  for i in range(len(d)):
    start = time.time()
    print("step {}".format(i))
    for epoch in range(num_epochs):
      out = net.forward(d[i], len(d[i]))

      future_index = math.ceil(i + (timePeriod/batchSize))
      if epoch == 0 and simulation_day == 0 and future_index < len(d):
        with torch.no_grad():
          future_index = math.ceil(i + (timePeriod/batchSize))
          sim_out = net.forward(d[future_index], len(d[future_index]))
          weights = sim_out[0].view(numAssets)
          percent_change = torch.dot(d.future_returns(future_index)[0], weights)
          overall_val *= 1 + percent_change
          print("return:",overall_val)
          print("allocs: ",weights)
      
      loss = loss_fn(out, i, len(d[i]), d.future_returns(i), timePeriod)
      losses_new_net.append(loss.item())
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    total_time += time.time() - start
    avg_time = total_time/(i + 1)
    print("eta: {}m {}s".format(int(avg_time/60 * (len(d) - i - 1)), int((avg_time *(len(d) - i - 1)) % 60 )))
    simulation_day += len(d[i])
    if simulation_day >= timePeriod:
      simulation_day = 0

  print(overall_val)
  return weights,net

def validation_set(testing_d,net,NUM_ASSETS,TIME_PERIOD_LENGTH):
  overall_val = 1
  simulation_day = 0
  x = [testing_d.dates[0][0][0][0]]
  y = [1]
  print(testing_d.dates[0])
  for i in range(len(testing_d)):
    with torch.no_grad():
      if simulation_day == 0:
        #print("input:", testing_d[i])
        out = net.forward(testing_d[i], len(testing_d[i]))
        #print(out)
        returns = testing_d.future_returns(i)[0]
        weights = out[0].view(NUM_ASSETS)
        print("weights:", weights)
        print("current:", testing_d.current_day_prices[i][0], "future:", testing_d.future_day_prices[i][0], "calculated change:", returns)
        percent_change = torch.dot(returns, weights)
        overall_val *= (1 + percent_change)
        print(overall_val)
        x.append(testing_d.dates[0][i][0][0])
        y.append(overall_val.item())
      
    simulation_day += len(testing_d[i])
    if simulation_day >= TIME_PERIOD_LENGTH:
      simulation_day = 0
      
  return x,y
  
def validation_set_earnings(testing_d,net,NUM_ASSETS,TIME_PERIOD_LENGTH):
  overall_val = 1
  simulation_day = 0
  x = [testing_d.dates[0][0][0][0]]
  y = [1]
  print(testing_d.dates[0])
  for i in range(len(testing_d)):
    with torch.no_grad():
      if simulation_day == 0:
        #print("input:", testing_d[i])
        out = net.forward(testing_d[i], testing_d.earnings[i], len(testing_d[i]))
        #print(out)
        returns = testing_d.future_returns(i)[0]
        weights = out[0].view(NUM_ASSETS)
        print("weights:", weights)
        print("current:", testing_d.current_day_prices[i][0], "future:", testing_d.future_day_prices[i][0], "calculated change:", returns)
        percent_change = torch.dot(returns, weights)
        overall_val *= (1 + percent_change)
        print(overall_val)
        x.append(testing_d.dates[0][i][0][0])
        y.append(overall_val.item())
      
    simulation_day += len(testing_d[i])
    if simulation_day >= TIME_PERIOD_LENGTH:
      simulation_day = 0
  return x,y

def train_net_earnings(d,timePeriod,numAssets,numFeatures,batchSize):
  #print(d)
  overall_val = 1
  start_day = 0
  net = NetWithEarnings(numFeatures,d.NUM_EARNINGS_FEATURES,numAssets,timePeriod).to('cuda')
  losses_new_net = []
  num_epochs = 100
  optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay = 0)
  loss_fn = sharpe_loss
  total_time = 0
  simulation_day = 0
  weights = []
  for i in range(len(d)):
    start = time.time()
    print("step {}".format(i))
    for epoch in range(num_epochs):
      out = net.forward(d[i], d.earnings[i], len(d[i]))

      future_index = math.ceil(i + (timePeriod/batchSize))
      if epoch == 0 and simulation_day == 0 and future_index < len(d):
        with torch.no_grad():
          future_index = math.ceil(i + (timePeriod/batchSize))
          sim_out = net.forward(d[future_index], d.earnings[future_index], len(d[future_index]))
          weights = sim_out[0].view(numAssets)
          percent_change = torch.dot(d.future_returns(future_index)[0], weights)
          overall_val *= 1 + percent_change
          print("return:",overall_val)
          print("allocs: ",weights)
      
      loss = loss_fn(out, i, len(d[i]), d.future_returns(i), timePeriod)
      losses_new_net.append(loss.item())
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    total_time += time.time() - start
    avg_time = total_time/(i + 1)
    print("eta: {}m {}s".format(int(avg_time/60 * (len(d) - i - 1)), int((avg_time *(len(d) - i - 1)) % 60 )))
    simulation_day += len(d[i])
    if simulation_day >= timePeriod:
      simulation_day = 0

  print(overall_val)
  return weights,net
