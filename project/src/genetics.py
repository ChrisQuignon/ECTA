#!/usr/bin/env python

from imp import load_source
from random import randrange, random, choice, gauss, uniform, sample
import numpy as np
import pylab
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from decisiontree import *
from copy import deepcopy
from datetime import datetime, timedelta
import time

helper = load_source('dsimport', 'helpers/helper.py')

ds = helper.dsimport()

df = pd.DataFrame(ds)
df.set_index(df.Date, inplace=True)
df = df.interpolate()
df.fillna(inplace=True, method='ffill')#we at first forwardfill
df.fillna(inplace=True, method='bfill')#then do a backwards fill
df = df.resample('5Min',how="mean")
df_norm = (df - df.mean()) / (df.max() - df.min())
# df = df_norm;
df_norm = df_norm.dropna();
train_start = df_norm.index.searchsorted(datetime(2014, 7, 1,0,0))
train_end = df_norm.index.searchsorted(datetime(2014, 12, 1,0,0))


train_data = df_norm.ix[train_start:train_end]
# to_be_predicted = ['Energie'];
# to_be_input = ["Aussentemperatur","Niederschlag","Relative Feuchte","Ruecklauftemperatur",
# 			   "Volumenstrom" , "Vorlauftemperatur"]
#
# tip = train_data[to_be_input]
# top = train_data[to_be_predicted]

class Genome():
    def __init__(self, data, genotype, predict_feat = 'Energie', leaf_mutation = 0.1, node_mutation = 0.8):
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
        return self.genotype.mse

    def crossover(self, partner):

        #TODO balance randomness between nodes and leafs

        # we take a random position
        own_idx = randrange(self.genotype.size())

        if partner.genotype.size() > 1:
            # we take a random subtree
            partner_idx = randrange(0, partner.genotype.size())
            partner_subtree = partner.genotype[partner_idx]
        else:#we mutate completely
            partner_subtree = partner.genotype

        if own_idx == 0:
            self.genotype = deepcopy(partner_subtree)
        else:
            self.genotype[own_idx] = deepcopy(partner_subtree)



class Evolution():
    def __init__(self, dataframe, init_tree_depth = 4, predict_feat='Energie', iterations=10, selection_type = '1+1', sigma = 0.002, ):

        self.df = dataframe
        self.iterations = iterations
        self.predict_feat = predict_feat
        self.init_tree_depth = init_tree_depth


        self.min_pred_val = min(self.df[self.predict_feat])
        self.max_pred_val = max(self.df[self.predict_feat])


        self.selection_type = [c for c in selection_type if not c.isdigit()][0]
        self.parents, self.offsprings = map(int, selection_type.split(self.selection_type))

        self.pop = []
        self.sigma = sigma

        self.best_fitness = []
        self.sigmas = []

        ##INITIALIZATION
        for _ in range(0, self.parents):
            tree = self.spawn_tree(init_tree_depth)
            genome = Genome(df, tree)
            self.pop.append(genome)

    def spawn_tree(self, depth):
        left = []
        right = []

        if depth == 1:
            leaf = uniform(self.min_pred_val, self.max_pred_val)
            leaf = DecisionLeaf(leaf)
            return leaf
        elif depth < 1:
            return
        else:
            left = self.spawn_tree(depth - 1)
            right = self.spawn_tree(depth - 1)

            feature = choice([ x for x in df.columns if x is not self.predict_feat])

            split = choice(self.df[feature])

        return DecisionTree(feature, split, left, right)


    def run(self):
        for i in range(self.iterations):

            start_time = time.time()
            self.evaluation()
            end_time = time.time()
            print("EVAL: %s seconds" %(end_time - start_time))

            self.selection()
            self.mutation()


            #self.crossover() happens in selection

            # print round(self.imp_rate, 2), 'succ with :', self.sigma
            # self.best_fitness.append(self.pop[0].fitness())
            # self.sigmas.append(self.sigma)
            # print map(lambda x: x.fitness(), self.pop)
            # print map(lambda x : x.genotype, self.pop)

    def evaluation(self):
        #who won?
        #TODO: change n_samples

        n_samples = 1000
        rows = np.random.choice(self.df.index.values, n_samples)

        #removing duplicates
        rows = list(rows)
        rows = [x for x in rows if not rows.count(x) > 1]

        samples = self.df.ix[rows]

        for g in self.pop:
            g.genotype.update_mse(samples, samples[self.predict_feat])

        self.pop = sorted(self.pop, key = lambda x: x.fitness)


    def selection(self):

        parents = [self.pop[i] for i in range(self.parents)]
        kids = []

        for _ in range(self.offsprings):

            kid = deepcopy(choice(parents))
            partner = choice(parents)
            kid.crossover(partner)
            kids.append(kid)

        if self.selection_type == '+':
            self.pop = parents + kids

        elif self.selection_type == ',':
            self.pop = kids
        else:
            print "selection type unkown!"

    def mutation(self):
        for g in self.pop:
            g.mutate()
        pass



e = Evolution(df, 3, selection_type = '1+1')
# e = Evolution(train_data)
e.run()
# print e.pop
# print "selecting"
# e.selection()
# print e.pop
# print 'crossover'
# print e.pop
#
#
# dT = DecisionTree('Aussentemperatur', 7, 7.7, 64.0)
# dN1 = DecisionTree('Vorlauftemperatur', 8.2, dT, 64.6)
# dN1.update_mse(df[0:100], df.Energie[0:100])
# print dN.predict(df[10:20])
#
# dT = DecisionTree('Aussentemperatur', 17, 17.7, 164.0)
# dN2 = DecisionTree('Vorlauftemperatur', 18.2, dT, 164.6)
# dN2.update_mse(df[0:100], df.Energie[0:100])
#
# g1 = Genome(df, dN1)
#
# g2 = Genome(df, dN2)
