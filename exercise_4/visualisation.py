#!/usr/bin/env python

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
from itertools import groupby

ds = []
keys = [
        "iterations" ,
        "selection_type",
        "sigma",
        "sigma_delta",
        "fitness_mean",
        "best_fitness"
    ]

#IMPORT FILE
input_file = csv.DictReader(open("data_vehicle_simfull_run.csv"))
for row in input_file:
    ds.append(row)

# print len(ds)

#value conversion
for d in ds:
    d['iterations'] = int(d['iterations'])
    d['sigma'] = float(d['sigma'])
    d['sigma_delta'] = float(d['sigma_delta'])
    d['fitness_mean'] = float(d['fitness_mean'])
    d['best_fitness'] = float(d['best_fitness'])



groups = []
uniquekeys = []
for k, g in groupby(ds, lambda x : x['selection_type']):
   groups.append(list(g))    # Store group iterator as a list
   uniquekeys.append(k)

# print groups
# print uniquekeys

print groups[0]

for ds in groups:

    print [d["fitness_mean"] for d in ds]

    # # PLOTS
    fig = plt.figure()
    ax = fig.add_subplot(111)# projection='3d'
    # m = [d["sigma"] for d in ds]
    # p = [d["select_perc"] for d in ds]
    # pop = [d["pop_size"] for d in ds]

    # #Fitness
    ax.plot([d["fitness_mean"] for d in ds], 'bx', ms = 10)
    ax.plot([d["best_fitness"] for d in ds], 'gx', ms = 10)
    plt.title(ds[0]['selection_type'])
    plt.ylabel("fitness")
    plt.xlabel("run")
    plt.ylim(0,1228892.1190044212*1.1)
    plt.show()

    plt.clf()

    # #SIGMA
    # ax.plot([d["sigma_delta"] for d in ds], [d["sigma"] for d in ds], 'bx', ms = 10)
    # # ax.plot([d["sigma_delta"] for d in ds], 'gx', ms = 10)
    # plt.ylabel("sigma")
    # plt.xlabel("sigma delta")
    # # plt.ylim(0,plt.ylim()[1])
    # plt.show()
