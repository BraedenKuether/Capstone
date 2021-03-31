import torch 

class Batcher:
  
  def __init__(self,dataset,bSize):
    self.batchSize = bSize
    self.dataset = dataset
    X,y = self.makeBatch(self.dataset)
    self.X = torch.split(X,self.batchSize)
    self.y = torch.split(y,self.batchSize) 

    if len(dataset) % bSize:
      self.X = self.X[:-1]
      self.y = self.y[:-1]

    self.size = len(self.X)
    
  def __len__(self):
    return self.size

  def makeBatch(self,d):
    xs = []
    ys = []
    for i in range(len(d)):
      X,y = d[i] 
      X = torch.unsqueeze(X,0)
      y = torch.unsqueeze(y,0)
      xs.append(X)
      ys.append(y)
    return torch.cat(xs,0),torch.cat(ys,0)

  def __getitem__(self,i):
    return self.X[i],self.y[i] 
      

