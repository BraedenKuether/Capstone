from Model import Net
import torch
from torch import optim
import math
import time

torch.set_default_tensor_type('torch.cuda.FloatTensor')
def sharpe_loss(weights, batch_pos, batch_len, returns):
  total_ratio = 0
  total_ratio = 0
  er = 0
  er2 = 0
  risk_free_rate = 0.09
  er_list = []
  for batch in range(batch_len):
    r_i_t = returns[batch]
    curr_er = torch.dot(r_i_t, weights[batch])
    er += curr_er
    er2 += curr_er**2
    er_list.append(curr_er)
  er_list = torch.Tensor(er_list)
  er = er/batch_len
  er2 = er2/batch_len
  ratio = (er-risk_free_rate) / torch.std(er_list)
  ratio = -1 * ratio
  return ratio
  
  
def train_net(d,timePeriod,numAssets,numFeatures,batchSize):
    overall_val = 1
    start_day = 0
    net = Net(numFeatures,numAssets,timePeriod).to('cuda')
    losses_new_net = []
    num_epochs = 100
    optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay = 0)
    loss_fn = sharpe_loss
    total_time = 0
    simulation_day = 0
    for i in range(len(d)):
      start = time.time()
      #print("step {}".format(i))
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
            #print("weights:", weights)
            #print("current:", d[i][0][-1].view(NUM_ASSETS, int(NUM_FEATURES/NUM_ASSETS))[:,0], "future:", d.future_returns(i)[0])
            print("return:",overall_val)
            print("allocs: ",weights)
        
        loss = loss_fn(out, i, len(d[i]), d.future_returns(i))
        losses_new_net.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      total_time += time.time() - start
      avg_time = total_time/(i + 1)
      print("eta: {}m {}s".format(int(avg_time/60 * (len(d) - i - 1)), int((avg_time * (len(d) - i - 1)) % 60 )))
      simulation_day += len(d[i])
      if simulation_day >= timePeriod:
        simulation_day = 0

    print(overall_val)
    return net
