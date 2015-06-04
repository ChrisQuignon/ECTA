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
import multiprocessing
from itertools import product
import csv

#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', 'helpers/helper.py')

ds = helper.dsimport()

# ds = helper.stretch(ds)

df = pd.DataFrame(ds)
df.set_index(df.Date, inplace = True)
df.Energie.resample('1Min', fill_method="ffill")
df = df.resample('1Min')
df.Energie.resample('D')
# df.interpolate(inplace=True)
df.fillna(inplace=True, method='ffill')#we at first forwardfill
# df.fillna(inplace=True, method='bfill')#then do a backwards fill

df_norm = df.dropna();
df_norm = (df_norm - df_norm.min())
df_norm = df_norm /df_norm.max()
# df = df_norm;
train_start = df_norm.index.searchsorted(datetime(2014, 7, 1,0,0))
train_end = df_norm.index.searchsorted(datetime(2014, 12, 1,0,0))


train_data = df_norm.ix[train_start:train_end]
test_data = df_norm.ix[train_end:]
test_data = test_data.resample('D')

class Genome():
    def __init__(self, data, genotype, sigma = 0.2, leaf_mutation = 0.1, node_mutation = 0.8, predict_feat = 'Energie'):
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

        self.sigma = sigma
        self.leaf_mutation = leaf_mutation
        self.node_mutation  = node_mutation


    def mutate(self):
        for subtree in self.genotype:

            #depth should not go bejond 30
            if subtree.depth() > 30:
                print "Tree depth of ", subtree.depth(), 'reached'
                return

            if isinstance(subtree, DecisionLeaf):
                if random() < self.leaf_mutation:
                    #TODO: change sigma
                    sigma = self.sigma * subtree.val
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
                    sigma = self.sigma * subtree.split
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

def par_mse(inp):
    g, samples, predict_feat = inp
    g.genotype.update_mse(samples, samples[predict_feat])
    return g

class Evolution():
    def __init__(self, dataframe, init_tree_depth = 4, predict_feat='Energie', iterations=100, selection_type = '1+1'):

        self.df = dataframe
        self.iterations = iterations
        self.predict_feat = predict_feat
        self.init_tree_depth = init_tree_depth


        self.min_pred_val = min(self.df[self.predict_feat])
        self.max_pred_val = max(self.df[self.predict_feat])


        self.selection_type = [c for c in selection_type if not c.isdigit()][0]
        self.parents, self.offsprings = map(int, selection_type.split(self.selection_type))

        self.pop = []
        self.best_fitness = []
        ##INITIALIZATION
        for _ in range(0, self.parents):
            tree = self.spawn_tree(self.init_tree_depth)
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

        mins = []
        means = []
        maxs = []

        self.evaluation()
        for i in range(self.iterations):

            # start_time = time.time()
            # end_time = time.time()
            # print("EVAL: %s seconds" %(end_time - start_time))

            self.selection()
            self.mutation()
            fits = self.evaluation()

            mins.append(fits[0])
            maxs.append(fits[-1])
            means.append(np.mean(fits))

        #AND THE WINNER IS
        winner = self.pop[0].genotype
        best_result = winner.predict(test_data)
        mse = winner.update_mse(test_data, test_data[self.predict_feat])

        #PLOT min man max plot
        # pylab.ylim((0,1))
        pylab.yscale('log')
        pylab.title(str(self.parents) + self.selection_type + str(self.offsprings) + ': ' + str(min(mins)))
        pylab.xlabel('iteration')
        pylab.ylabel('fitness')
        pylab.plot(mins, color = 'green')
        pylab.plot(maxs, color = 'red')
        pylab.plot(means, color = 'blue')
        pylab.tight_layout()
        pylab.savefig('img/' + str(min(mins)) + '-' + str(self.parents) + self.selection_type + str(self.offsprings) + '.png')
        # pylab.show()
        pylab.clf()

        #PLOT PREDICTION
        # pylab.ylim((0,1))
        pylab.yscale('log')
        pylab.title(str(self.parents) + self.selection_type + str(self.offsprings) + ': ' + str(min(mins)))
        pylab.xlabel('date')
        pylab.ylabel('fitness')
        # pylab.plot_date(x=test_data[self.predict_feat].index, y=best_result, fmt="r-")
        # pylab.plot_date(x=test_data[self.predict_feat].index, y=test_data[self.predict_feat], fmt="g-")
        pylab.plot(best_result, "r-")
        pylab.plot(test_data[self.predict_feat], "g-")
        pylab.tight_layout()
        # pylab.show()
        pylab.savefig('img/best-' + str(winner.mse) + '-' + str(self.parents) + self.selection_type + str(self.offsprings) + '.png')
        pylab.clf()

        return min(mins)

    def evaluation(self):
        #TODO: change n_samples
        n_samples = 100
        rows = np.random.choice(self.df.index.values, n_samples)

        #removing duplicates
        rows = list(rows)
        rows = [x for x in rows if not rows.count(x) > 1]

        samples = self.df.ix[rows]

        #seriell
        for g in self.pop:
            g.genotype.update_mse(samples, samples[self.predict_feat])

        # # parallel
        # # drive access is not faster...
        # inps = [(p, samples, self.predict_feat) for p in self.pop]
        # pool = multiprocessing.Pool(len(self.pop))
        # self.pop = pool.map(par_mse, inps)
        # pool.close()
        # pool.join()

        self.pop = sorted(self.pop, key = lambda x : x.fitness())

        # print 'POP'
        # for p in self.pop:
        #     print 'mse:', "%0.4f" %(p.fitness()), ' size:', p.genotype.size()
        #     # print p.genotype
        # print ''
        return [x.fitness() for x in self.pop]


    def selection(self):
        self.pop = [p for p in self.pop if not np.isnan(p.fitness()) ]#drop nans
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
        for g in self.pop[1:]:#ELITISM
            g.mutate()
        pass


def par_wrap(arg):
    sigma, iterations, selection, init_tree_depth, leaf_mutation, node_mutation = arg

    print 'start', arg

    e = Evolution(train_data,
                    iterations = iterations,
                    init_tree_depth=init_tree_depth,
                    predict_feat='Energie',
                    selection_type = selection)
    for genome in e.pop:
        genome.sigma = sigma
        genome.leaf_mutation = leaf_mutation
        genome.node_mutation = node_mutation
    min_ = e.run()

    d = {"sigma"           : sigma,
         "iterations"      : iterations,
         "selection"       : selection,
         "init_tree_depth" : init_tree_depth,
         "leaf_mutation"   : leaf_mutation,
         "node_mutation"   : node_mutation,
         'min'             : min_,
         'tree'            : e.pop[0].genotype
         }

    print 'end', arg
    return d

sigmas = [0.4, 0.2, 0.1, 0.05, 0.005]#
iterations = [100]
selections = ['2+2',  '4+16', '2,2', '4,16']#, '1,10', '2,2', '2,20', '4,4', '10,10']
init_tree_depths = [3, 6]
# leaf_mutations = [0.4, 0.2, 0.1, 0.01]
# node_mutations = [0.8, 0.6, 0.4, 0.2]
#
# sigmas = [0.2]
# iterations = [100]
# selections = ['1+4']
# init_tree_depths = [2]
leaf_mutations = [0.1]
node_mutations = [0.8]

args = [sigmas, iterations, selections, init_tree_depths, leaf_mutations, node_mutations]

args = list(product(*args))

ds = []


# # parallel run
pool = multiprocessing.Pool(4)
ds = pool.map(par_wrap, args)
pool.close()
pool.join()

keys = ["sigma",
     "iterations",
     "selection",
     "init_tree_depth",
     "leaf_mutation",
     "node_mutation",
     'min',
     'tree'
]

with open('img/treerun.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(ds)




# e = Evolution(train_data, 4, iterations = 10, selection_type = '2+2')
# e.run()
