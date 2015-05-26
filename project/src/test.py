#!/usr/bin/env python

from imp import load_source
from random import randrange, random
import numpy as np
from datetime import datetime, timedelta
import pylab
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import csv

from itertools import permutations

#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', 'helpers/helper.py')

df = pd.read_csv('data/ds1_weather.csv', decimal=',',sep=';', parse_dates=True, index_col=[0])

df.interpolate(inplace=True)
df = df.resample('5Min')
df.fillna(inplace=True, method='ffill')#we at first forwardfill
df.fillna(inplace=True, method='bfill')#then do a backwards fill

labels = [l for l in df]
#
# #PLOT 2 FEATURES, COLORED BY TIME
# for xlabel, ylabel in permutations(labels, 2):
#     pylab.figure(figsize=(20,10))
#     sc = pylab.scatter(df[xlabel], df[ylabel], c = [t.hour for t in df.index], cmap='cool', alpha=0.3,  lw = 0)
#     pylab.xlabel(helper.translate(xlabel))
#     pylab.ylabel(helper.translate(ylabel))
#     cbar = pylab.colorbar(sc)
#     cbar.set_label( 'h of the day')
#
#     pylab.savefig('img/plot-' + helper.translate(xlabel)+ '-' + helper.translate(ylabel) +  '.png')
#     # pylab.show()
#     pylab.clf()


#PLOT ONE FEATURE TO TIME, COLORED BY ENERGY
for label in labels:
    pylab.figure(figsize=(20,10))
    sc = pylab.scatter([t.minute + t.hour*60 for t in df.index], df[label], c = df.Energie, cmap='cool', alpha=0.3,  lw = 0)
    cbar = pylab.colorbar(sc)
    cbar.set_label( 'energy')
    pylab.ylabel(helper.translate(label))
    pylab.xlabel('min of the day')
    # pylab.tight_layout()


    pylab.savefig('img/time-energy-' + helper.translate(label)+ '-h.png')
    # pylab.show()
    pylab.clf()
