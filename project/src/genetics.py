#!/usr/bin/env python

from imp import load_source
from random import randrange, random
import numpy as np
import pylab
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from decisiontree import *

#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', 'helpers/helper.py')

df = pd.read_csv('data/ds1_weather.csv', decimal=',',sep=';', parse_dates=True, index_col=[0])


df.interpolate(inplace=True)
df = df.resample('1Min')
df.fillna(inplace=True, method='ffill')#we at first forwardfill
df.fillna(inplace=True, method='bfill')#then do a backwards fill


dT = decisionTree('Aussentemperatur', 7, 7.7, 64.0)
dN = decisionTree('Vorlauftemperatur', 8.2, dT, 64.6)
dN.update_mse(df[0:100], df.Energie[0:100])
print dN.predict(df[10:20])
