import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class PortfolioDataSet(Dataset):
  # assumes 5 year history for all assets need to generalize eventually
  def __init__(self,combined_data,TIME_PERIOD_LENGTH,NUM_ASSETS,NUM_FEATURES,BATCH_SIZE):
    self.raw = combined_data
    self.n = NUM_ASSETS
    self.window = TIME_PERIOD_LENGTH
    self.features = NUM_FEATURES
    self.batch_size = BATCH_SIZE
    self.data = []
    self.non_normal_data = []
    self.returns = []
    self.future_day_prices = []
    self.current_day_prices = []
    scaler = MinMaxScaler()
    i = 0
    while i + 2*self.window - 1 < len(self.raw):
      select = self.raw[i:i+self.window].tolist() # get a TIME_PERIOD_LENGTH chunk
      future_day_prices = self.raw[i + 2*self.window - 1].view((self.n,int(self.features/self.n)))[:,0] # get day close 
      last_day_prices = self.raw[i + self.window - 1].view((self.n,int(self.features/self.n)))[:,0] 
      future_returns = future_day_prices/last_day_prices - 1
      scaler.fit(select)
      normalized = scaler.transform(select)
      self.non_normal_data.append(select)
      self.data.append(normalized)
      self.returns.append(future_returns.tolist())
      self.current_day_prices.append(last_day_prices.tolist())
      self.future_day_prices.append(future_day_prices.tolist())
      i += 1

    test_split = torch.split(torch.Tensor(self.non_normal_data), BATCH_SIZE)
    if len(test_split[-1]) == 1:
      self.non_normal_data = self.non_normal_data[1:]
      self.data = self.data[1:]
      self.returns = self.returns[1:]
      self.future_day_prices = self.future_day_prices[1:]
      self.current_day_prices = self.current_day_prices[1:]


    #print(torch.Tensor(self.data).shape)
    self.non_normal_data = torch.split(torch.Tensor(self.non_normal_data),BATCH_SIZE)
    self.data = torch.split(torch.Tensor(self.data),BATCH_SIZE)
    self.returns = torch.split(torch.Tensor(self.returns),BATCH_SIZE)
    self.future_day_prices = torch.split(torch.Tensor(self.future_day_prices),BATCH_SIZE)
    self.current_day_prices = torch.split(torch.Tensor(self.current_day_prices),BATCH_SIZE)
    
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
    
