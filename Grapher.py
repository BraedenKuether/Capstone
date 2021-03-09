import json

class DataPoint:
  def __init__(self,x,y):
    self.x = x
    self.y = y

class LineItem:
  def __init__(self,id,data):
    self.id = id
    self.data = data
    self.color = "hsl(198, 70%, 50%)"

  
def toLine(dataDict):
  #dict from names to data
  items = []

  for name in dataDict.keys():
    points = [DataPoint(x,y).__dict__ for x,y in dataDict[name]] 
    items.append(LineItem(name,points).__dict__)
  
  return json.dumps(items)





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
