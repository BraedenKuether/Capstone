import pyEX as px
import pandas as pd
import matplotlib.pyplot as plt

import Portfolio as p

class Tester:

    def __init__(self,P, fMaps):
        self.portfolio = P
        self.functions = fMaps

    def apply(self,Name):
        self.functions[fName](self.portfolio)

    def plotPortfolio(self,key="close"):
        plot = plt.gca()

        for asset in self.portfolio.assetsByTime:
            asset.plot(y=key,ax=plot)            

        plot.legend(self.portfolio.symbols)
        plot.set_title("Daily "+key)
        plt.show()


client = px.Client(version="sandbox")
stonks = ['vti', 'agg', 'dbc', 'vixy']

p = p.Portfolio(stonks,client)

ts = Tester(p,None)

ts.plotPortfolio("close")
ts.plotPortfolio("fVolume")

