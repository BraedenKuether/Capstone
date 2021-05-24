import json

'''
These classes and functions are for generating the json
objects needed for graphs if you want to understand them 
the best way is to just look at the data tab on

https://nivo.rocks/bar/
https://nivo.rocks/line/

The rest is just code to coax python into doing all the json
printing for us

'''

class DataPoint:
  def __init__(self,x,y):
    self.x = x
    self.y = y

class LineItem:
  def __init__(self,id,data):
    self.id = id
    self.data = data
    self.color = "hsl(198, 70%, 50%)"

class Bar:
  def __init__(self,id,data):
    self.ticker = id
    self.returns = data
    self.returnsColor = "hsl(198, 70%, 50%)"

def toLine(dataDict):
  '''
    creates line graph JSON for nivo
    
    params:
      dataDict: dictionary mapping the x value names to list of y values
    
    returns:
      items: list of json objects
  '''
  items = []

  for name in dataDict.keys():
    points = [DataPoint(x,y).__dict__ for x,y in dataDict[name]] 
    items.append(LineItem(name,points).__dict__)
  
  return items

def toBar(dataDict):
  '''
    creates bar graph JSON for nivo
    
    params:
      dataDict: dictionary mapping the x value names to list of y values
    
    returns:
      items: list of json objects
  '''
  items = []
  for name in dataDict.keys():
    point = Bar(name,dataDict[name]).__dict__
    items.append(point)
  
  return items


def weightsToJson(weights):
  for i in range(len(weights)):
    weights[i] = round(weights[i],4)
  data = {"weights":weights}
  return data



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
