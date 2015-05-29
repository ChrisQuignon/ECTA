#!/usr/bin/env python

from imp import load_source
from random import randrange, random, choice, gauss
import numpy as np
import pylab
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from decisiontree import *
from copy import deepcopy

#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', 'helpers/helper.py')

df = pd.read_csv('data/ds1_weather.csv', decimal=',',sep=';', parse_dates=True, index_col=[0])


df.interpolate(inplace=True)
df = df.resample('1Min')
df.fillna(inplace=True, method='ffill')#we at first forwardfill
df.fillna(inplace=True, method='bfill')#then do a backwards fill


class Genome():
    def __init__(self, data, predict_feat, genotype, leaf_mutation = 0.1, node_mutation = 0.8):
        if predict_feat in data.columns:
            self.df = data
            self.predict_feat = predict_feat
        else:
            print 'Feature ', predict_feat, ' could not be found.'
            self.predict_feat = data.columns[0]
            print 'Feature ', self.predict_feat, ' choosen instead.'

        self.predict_max = self.df[self.predict_feat].max()
        self.predict_min = self.df[self.predict_feat].min()

        if isinstance(genotype, DecisionTree):
            self.genotype = genotype
        else:
            print "Genotype ", genotype, 'unknown'
            print "Random leaf set"

            feature  = choice(self.df.columns)
            self.genotype = DecisionLeaf(0.0)

        self.leaf_mutation = leaf_mutation
        self.node_mutation  = node_mutation


    def mutate(self):
        for subtree in self.genotype:

            if isinstance(subtree, DecisionLeaf):
                if random() < self.leaf_mutation:
                    #TODO: change sigma
                    sigma = 0.2 * subtree.val
                    delta = gauss(subtree.val, sigma)

                    new_val = subtree.val + delta

                    max_val = self.predict_max
                    min_val = self.predict_min

                    if new_val > max_val:
                        subtree.val = min_val + (max_val -  new_val)
                    elif new_val < min_val:
                        subtree.val = max_val - (min_val -  new_val)
                    else:
                        subtree.val = new_val


            elif isinstance(subtree, DecisionTree):
                if random() < self.node_mutation:

                    #TODO: change sigma
                    sigma = 0.2 * subtree.split
                    delta = gauss(subtree.split, sigma)

                    new_val = subtree.split + delta
                    #TODO: fix performance
                    max_val = self.df[subtree.feature].max()
                    min_val = self.df[subtree.feature].min()

                    if new_val > max_val:
                        subtree.split = min_val + (max_val -  new_val)
                    elif new_val < min_val:
                        subtree.split = max_val - (min_val -  new_val)
                    else:
                        subtree.split = new_val


    def fitness(self):
        pass

    def crossover(self, partner):

        #TODO balance randomnessbetween nodes and leafs

        # we take a random position
        own_idx = randrange(self.genotype.size())

        # we take a random subtree
        partner_idx = randrange(1, partner.genotype.size())
        partner_subtree = partner.genotype[partner_idx]


        if own_idx == 0:
            self.genotype = deepcopy(partner_subtree)
        else:
            self.genotype[own_idx] = deepcopy(partner_subtree)



dT = DecisionTree('Aussentemperatur', 7, 7.7, 64.0)
dN1 = DecisionTree('Vorlauftemperatur', 8.2, dT, 64.6)
dN1.update_mse(df[0:100], df.Energie[0:100])
print dN.predict(df[10:20])

dT = DecisionTree('Aussentemperatur', 17, 17.7, 164.0)
dN2 = DecisionTree('Vorlauftemperatur', 18.2, dT, 164.6)
dN2.update_mse(df[0:100], df.Energie[0:100])

g1 = Genome(df, 'Energie', dN1)

g2 = Genome(df, 'Energie', dN2)

print g1.genotype
print '__'
print g2.genotype
print 'CROSSOVER'


g2.crossover(g1)

print g2.genotype

g2.mutate()
g2.genotype.update_mse(df[0:100], df.Energie[0:100])
g2.genotype.mse
