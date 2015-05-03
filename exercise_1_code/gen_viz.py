#!/usr/bin/env python

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
from itertools import groupby

ds = []
keys = [
    'min',
    'mean',
    'max',
    'ff',
    'opt',
    'max_iterations',
    'pop_size',
    'select_perc',
    'crossover_type',
    'mutation_prob']

#IMPORT FILE
input_file = csv.DictReader(open("data_genetic.csv"))
for row in input_file:
    ds.append(row)

crossovers = ['onepoint', 'twopoint', 'threepoint', 'mean', 'fitnessmean', 'randomweight']
ffs = ['Plateau3D', 'SquaredError2D', 'Trimodal2D']
opts = ['Genetic']

print len(ds)


for d in ds:
  d['min'] = float(d['min'])
  d['max'] = float(d['max'])
  d['mean'] = float(d['mean'])
  d['max_iterations'] = int(d['max_iterations'])
  d['pop_size'] = int(d['pop_size'])
  d['select_perc'] = float(d['select_perc'])
  d['mutation_prob'] = float(d['mutation_prob'])

#for d in ds:
#    for k in d:
#        print type(d[k]), d[k]
folder = 'analysis/'
#ds = sorted(ds, key=lambda k: k['max'])
for c in crossovers:
    for ff in ffs:

        print c
        print ff

        group = [d for d in ds if d['crossover_type'] == c and d['ff'] == ff]

        _min = [d['min'] for d in group ]
        _mean = [d['mean'] for d in group ]
        _max = [d['max'] for d in group ]

        _pop = [d['pop_size'] for d in group ]
        _mut = [d['mutation_prob'] for d in group ]
        _select = [d['select_perc'] for d in group ]

        # print len

        #MIN MEAN MAX
        fig = plt.figure()
        ax = fig.add_subplot(111)# projection='3d'
        # plt.ylim([0.5, 2.5])



        plt.scatter( _pop, _min, c = 'blue', marker = 'x', alpha = 0.6, s = 100, cmap=plt.cm.coolwarm)
        plt.scatter( _pop, _mean, c = 'green', marker = 'o', linewidth='0', alpha = 0.6, s = 100,  cmap=plt.cm.coolwarm)
        plt.scatter( _pop, _max, c = 'red', marker = 'x', alpha = 0.6, s = 100,  cmap=plt.cm.coolwarm)
        plt.title('Crossover type: ' + c)
        plt.ylabel('Fitness on landscape ' + ff)
        plt.xlabel('Population size')
        plt.savefig(folder + c + "-" + ff + "-pop" + '.png')
        # plt.show()

        plt.scatter( _mut, _min, c = 'blue', marker = 'x', alpha = 0.6, s = 100,  cmap=plt.cm.coolwarm)
        plt.scatter( _mut, _mean, c = 'green', marker = 'o', linewidth='0', alpha = 0.6, s = 100,  cmap=plt.cm.coolwarm)
        plt.scatter( _mut, _max, c = 'red', marker = 'x', alpha = 0.6, s = 100,  cmap=plt.cm.coolwarm)
        plt.title('Crossover type: ' + c)
        plt.ylabel('Fitness on landscape ' + ff)
        plt.xlabel('Mutation rate')
        plt.savefig(folder + c + "-" + ff + "-mut" + '.png')
        # plt.show()

        plt.scatter( _select, _min, c = 'blue', marker = 'x', alpha = 0.6, s = 100,  cmap=plt.cm.coolwarm)
        plt.scatter( _select, _mean, c = 'green', marker = 'o', linewidth='0', alpha = 0.6, s = 100,  cmap=plt.cm.coolwarm)
        plt.scatter( _select, _max, c = 'red', marker = 'x', alpha = 0.6, s = 100,  cmap=plt.cm.coolwarm)
        plt.title('Crossover type: ' + c)
        plt.ylabel('Fitness on landscape ' + ff)
        plt.xlabel('Selection rate')
        plt.savefig(folder + c + "-" + ff + "-select" + '.png')
        # plt.show()
