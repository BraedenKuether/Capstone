# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 23:27:39 2021

@author: Ash
"""
import pyEX as p

IEX_TOKEN = "Tpk_9239047b3f394b3d83a80bda63b3d061"
client = p.Client(api_token = IEX_TOKEN, version='sandbox')
df = client.balanceSheetDF('AAPL', period='quarter', last = 12)