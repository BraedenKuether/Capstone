import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import numpy as np
import datetime
import random
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class PortfolioDataSet(Dataset):
  # assumes 5 year history for all assets need to generalize eventually
  def __init__(self,combined_data,dates,TIME_PERIOD_LENGTH,NUM_ASSETS,NUM_FEATURES,BATCH_SIZE,earnings = None, testing = False, test_length = 126, num_earning_feats = None):
    self.raw = combined_data
    if not testing:
      if earnings:
        self.testing_set = PortfolioDataSet(self.raw[-test_length:], dates[-test_length:],TIME_PERIOD_LENGTH,NUM_ASSETS,NUM_FEATURES,1,testing = True, earnings = earnings)
      else:
        self.testing_set = PortfolioDataSet(self.raw[-test_length:], dates[-test_length:],TIME_PERIOD_LENGTH,NUM_ASSETS,NUM_FEATURES,1,testing = True)
      self.raw = self.raw[:(-test_length - TIME_PERIOD_LENGTH)]
      dates = dates[:(-test_length - TIME_PERIOD_LENGTH)]
    self.n = NUM_ASSETS
    self.window = TIME_PERIOD_LENGTH
    self.features = NUM_FEATURES
    self.batch_size = BATCH_SIZE
    self.data = []
    self.non_normal_data = []
    self.returns = []
    self.future_day_prices = []
    self.current_day_prices = []
    self.dates = []
    self.future_dates = []
    if earnings:
      self.NUM_EARNINGS_FEATURES = num_earning_feats
      for i in range(len(earnings)):
        earnings[i] = earnings[i].sort_index()
      raw_earnings = earnings.copy()
      self.earnings = []
    scaler = MinMaxScaler()
    i = 0
    print("len raw:",len(self.raw))
    while i + 2*self.window < len(self.raw):
      select = self.raw[i:i+self.window].tolist() # get a TIME_PERIOD_LENGTH chunk
      #for each day, gets the percent change time_period_length days in the future
      #print(self.features, self.n)
      future_day_prices = [[day[j] for j in range(0, self.features, int(self.features/self.n))] for day in self.raw[i + self.window : i + 2*self.window]] # get day close 
      last_day_prices = [[day[j] for j in range(0, self.features, int(self.features/self.n))] for day in self.raw[i : i + self.window]]
      future_returns = torch.tensor(future_day_prices)/torch.tensor(last_day_prices) - 1
      scaler.fit(select)
      normalized = scaler.transform(select)
      self.non_normal_data.append(select)
      self.data.append(normalized)
      self.returns.append(future_returns.tolist())
      self.current_day_prices.append(last_day_prices)
      self.future_day_prices.append(future_day_prices)
      self.dates.append(dates[i:i+self.window].tolist())
      self.future_dates.append(dates[i + self.window : i + 2*self.window].tolist())
      if earnings:
        last_date = datetime.datetime.strptime(dates[i + self.window - 1][0], '%Y-%m-%d')
        earnings_list = []
        for earning in raw_earnings:
          latest_earning = earning.loc[:last_date].iloc[-1]
          earnings_list.extend(latest_earning.tolist())
        self.earnings.append(earnings_list)
      i += 1

    test_split = torch.split(torch.Tensor(self.non_normal_data), BATCH_SIZE)
    if len(test_split[-1]) == 1:
      self.non_normal_data = self.non_normal_data[1:]
      self.data = self.data[1:]
      self.returns = self.returns[1:]
      self.future_day_prices = self.future_day_prices[1:]
      self.current_day_prices = self.current_day_prices[1:]
      self.dates = self.dates[1:]
      self.future_dates = self.future_dates[1:]
      if earnings:
        self.earnings = self.earnings[1:]


    #print(torch.Tensor(self.data).shape)
    self.non_normal_data = torch.split(torch.Tensor(self.non_normal_data),BATCH_SIZE)
    self.data = torch.split(torch.Tensor(self.data),BATCH_SIZE)
    self.returns = torch.split(torch.Tensor(self.returns),BATCH_SIZE)
    self.future_day_prices = torch.split(torch.Tensor(self.future_day_prices),BATCH_SIZE)
    self.current_day_prices = torch.split(torch.Tensor(self.current_day_prices),BATCH_SIZE)
    self.dates = [self.dates[i:i+BATCH_SIZE] for i in range(0,len(self.dates),BATCH_SIZE)]
    self.future_dates = [self.future_dates[i:i+BATCH_SIZE] for i in range(0,len(self.future_dates),BATCH_SIZE)]
    if earnings:
      self.earnings = torch.split(torch.Tensor(self.earnings),BATCH_SIZE)
    
  def __len__(self):
    return len(self.data)
  
  def future_returns(self, idx):
    return self.returns[idx]

  def return_size(self):
    return len(self.returns)
    
  def __getitem__(self,idx):
    return self.data[idx]
    
  def non_normal(self):
    return self.non_normal_data

  def shuffle(self):
    temp = list(zip(self.data, self.non_normal_data)) 
    random.shuffle(temp) 
    temp_data, temp_non_normal = zip(*temp)
    self.data,self.temp_non_normal = list(temp_data),list(temp_non_normal)
    
