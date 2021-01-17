import pyEX as p
class Portfolio:
    def __init__(self,assets,client):
        self.symbols = assets
        self.rawAssets = []
        self.assetsByTime = []
        self.numAssets = len(assets)
        self.client = client 

        for a in assets:
            self.assetsByTime.append(self.stockDF(self.client,a,0))

        self.rawAssets = self.toRaw(self.assetsByTime)

    def printAssets(self):
        for x in self.assetsByTime:
            print(x[:10])

    def stockDF(self,client, symb, interval):
        #the time of data appears to be inconsistent
        #may need to check this down the road
        return client.chartDF(symb,timeframe='5y')
    
    def toRaw(self,assets):
        feats = ['close','open','high','volume',\
             'uClose','uHigh','uLow','uVolume',\
             'fOpen','fClose','fHigh','fLow','fVolume']
        FV = []
        for a in assets:
            vals = []
            for f in feats:
                vals.append(list(reversed(a[f].values)))
            FV.append(vals)
        return FV
 

    def batch(self):
        pass             


client = p.Client(version="sandbox")
stonks = ['vti', 'agg', 'dbc', 'vixy']

p = Portfolio(stonks,client)
p.printAssets()





