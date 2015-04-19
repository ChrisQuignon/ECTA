#!/usr/bin/env python

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np

ds = []
keys = ['runs',
        'iterations',
        'pop_size',
        'select_perc',
        'mutation_prob',
        'min',
        'max',
        'mean']

#IMPORT FILE
input_file = csv.DictReader(open("data_tsp.csv"))
for row in input_file:
    ds.append(row)

# print len(ds)

#value conversion
for d in ds:
    d['runs'] = int(d['runs'])
    d['iterations'] = int(d['iterations'])
    d['pop_size'] = int(d['pop_size'])

    d['select_perc'] = float(d['select_perc'])
    d['mutation_prob'] = float(d['mutation_prob'])
    d['min'] = float(d['min'])
    d['max'] = float(d['max'])
    d['mean'] = float(d['mean'])


#MIN MEAN MAX
fig = plt.figure()
ax = fig.add_subplot(111)# projection='3d'
m = [d["mutation_prob"] for d in ds]
p = [d["select_perc"] for d in ds]
pop = [d["pop_size"] for d in ds]


plt.plot(p, [d["max"] for d in ds], 'rx', ms = 10)
plt.plot(p, [d["mean"] for d in ds], 'bx', ms = 10)
plt.plot(p, [d["min"] for d in ds], 'gx', ms = 10)
plt.ylabel("fitness")
plt.xlabel("percent selection")
plt.xlim(0.0, 0.7)
plt.show()

plt.plot(pop, [d["max"] for d in ds], 'rx', ms = 10)
plt.plot(pop, [d["mean"] for d in ds], 'bx', ms = 10)
plt.plot(pop, [d["min"] for d in ds], 'gx', ms = 10)
plt.ylabel("fitness")
plt.xlabel("population size")
plt.xlim(0.0, 210)
plt.show()

plt.plot(m, [d["max"] for d in ds], 'rx', ms = 10)
plt.plot(m, [d["mean"] for d in ds], 'bx', ms = 10)
plt.plot(m, [d["min"] for d in ds], 'gx', ms = 10)
plt.ylabel("fitness")
plt.xlabel("mutation probability")
plt.xlim(0.0, 0.4)
plt.show()
