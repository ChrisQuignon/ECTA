#!/usr/bin/env python

from imp import load_source
from random import randrange, random, choice, gauss, uniform, sample, randint
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
# helper = load_source('dsimport', 'helpers/helper.py')
#
# ds = helper.dsimport()
#
# # ds = helper.stretch(ds)
#
# df = pd.DataFrame(ds)
# df.set_index(df.Date, inplace = True)
# df.Energie.resample('1Min', fill_method="ffill")
# df = df.resample('1Min')
# df.Energie.resample('D')
# # df.interpolate(inplace=True)
# df.fillna(inplace=True, method='ffill')#we at first forwardfill
# # df.fillna(inplace=True, method='bfill')#then do a backwards fill
#
# df_norm = df.dropna();
# df_norm = (df_norm - df_norm.min())
# df_norm = df_norm /df_norm.max()
# # df = df_norm;
# train_start = df_norm.index.searchsorted(datetime(2014, 7, 1,0,0))
# train_end = df_norm.index.searchsorted(datetime(2014, 12, 1,0,0))
#
#
# train_data = df_norm.ix[train_start:train_end]
# test_data = df_norm.ix[train_end:]
# test_data = test_data.resample('D')

#
# inputs = np.loadtxt('data/inputs.txt')
# outputs = np.loadtxt('data/outputs.txt')
# val_in = np.loadtxt('data/val_in.txt')
# val_out = np.loadtxt('data/val_out.txt')

inputs = np.loadtxt('data/inputs_2024.txt')
outputs = np.loadtxt('data/outputs_2024.txt')
val_in = np.loadtxt('data/val_in_2024.txt')
val_out = np.loadtxt('data/val_out_2024.txt')

def bound(value, low=0, high=1):
    diff = high - low
    return (((value - low) % diff) + low)

class Genome():
    def __init__(self, inputs, outputs, genotype, sigma = 0.2, leaf_mutation = 0.1, node_mutation = 0.8):
        # if predict_feat in data.columns:
        self.inputs = inputs
        self.outputs = outputs
        # else:
        #     print 'Feature ', predict_feat, ' could not be found.'
        #     self.predict_feat = data.columns[0]
        #     print 'Feature ', self.predict_feat, ' choosen instead.'

        self.predict_max = max(outputs)
        self.predict_min = min(outputs)

        if isinstance(genotype, DecisionTree):
            self.genotype = genotype
        else:
            print "Genotype ", genotype, 'unknown'
            print "Random leaf set"

            feature  = choice(range(self.inputs.shape[0]))
            self.genotype = DecisionLeaf(0.5)

        self.sigma = sigma
        self.leaf_mutation = leaf_mutation
        self.node_mutation  = node_mutation


    def mutate(self):

        # TODO:
        #MAYBE NOT MUTATE EVERY SUBTREE!

        # for subtree in p.genotype:
        # for i in range(subtree.size()):
        #     if isinstance(subtree, DecisionTree):
        #         print subtree[i]

        # decide whether to mutate leaf or node
        # pick which one
        #   mutate
        s = self.genotype.size()

        # start_time = time.time()
        # nodes = [i for i in range(s) if isinstance(self.genotype[i], DecisionTree)]
        # leafs = [i for i in range(s) if isinstance(self.genotype[i], DecisionLeaf)]
        # end_time = time.time()
        # print("IDX: %s seconds" %(end_time - start_time))
        # print leafs
        # print nodes
        pos = choice(range(s))
        subtree = self.genotype[pos]

        # subtree = self.genotype[choice()]

        if isinstance(subtree, DecisionLeaf):#  and (random() < self.leaf_mutation)
            # pos = choice(leafs)
            # print 'leaf mutation, ', pos
            sigma = self.sigma * subtree.val
            delta = gauss(subtree.val, sigma)

            new_val = subtree.val + delta

            max_val = self.predict_max
            min_val = self.predict_min

            new_val = bound(new_val, self.predict_min, self.predict_max)

            subtree.val = new_val

        if isinstance(subtree, DecisionTree): #(random() < self.node_mutation) and
            # pos = choice(nodes)
            # print 'node mutation, ', pos
            sigma = self.sigma * subtree.split
            delta = gauss(subtree.split, sigma)

            new_val = subtree.split + delta

            max_val = max(self.inputs[subtree.feature])
            min_val = min(self.inputs[subtree.feature])

            new_val = bound(new_val, self.predict_min, self.predict_max)

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
    def __init__(self, inputs, outputs, init_tree_depth = 4,  iterations=100, selection_type = '1+1', n_samples = 100):
        self.inputs = inputs
        # print inputs
        self.outputs = outputs
        # self.df = dataframe
        self.iterations = iterations
        # self.predict_feat = predict_feat
        self.init_tree_depth = init_tree_depth
        self.n_samples = n_samples


        self.min_pred_val = min(self.outputs)
        self.max_pred_val = max(self.outputs)


        self.selection_type = [c for c in selection_type if not c.isdigit()][0]
        self.parents, self.offsprings = map(int, selection_type.split(self.selection_type))

        self.pop = []
        self.best_fitness = []
        ##INITIALIZATION
        for _ in range(0, self.parents):
            tree = self.spawn_tree(self.init_tree_depth)
            genome = Genome(self.inputs, self.outputs, tree)
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


            feature = choice(range(self.inputs.shape[1]))

            split = choice(self.inputs[feature])

            if split == 0.0:#zero splits are hard to mutate
                split = 0.5

        return DecisionTree(feature, split, left, right)


    def run(self):

        mins = []
        means = []
        maxs = []

        self.evaluation()
        for i in range(self.iterations):

            # start_time = time.time()
            self.selection()
            # end_time = time.time()
            # print("SLCT: %s seconds" %(end_time - start_time))

            # start_time = time.time()
            self.mutation()
            # end_time = time.time()
            # print("MUTN: %s seconds" %(end_time - start_time))

            # start_time = time.time()
            fits = self.evaluation()

            mins.append(fits[0])
            maxs.append(fits[-1])
            means.append(np.mean(fits))
            print i, ' of ', self.iterations

        #AND THE WINNER IS
        winner = self.pop[0].genotype

        #GLOBALS !
        best_result = winner.predict(val_in)
        mse = winner.update_mse(val_in, val_out)

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
        pylab.ylim((0,1))
        # pylab.yscale('log')
        pylab.title(str(self.parents) + self.selection_type + str(self.offsprings) + ': ' + str(min(mins)))
        pylab.xlabel('date')
        pylab.ylabel('prediction')
        # pylab.plot_date(x=test_data[self.predict_feat].index, y=best_result, fmt="r-")
        # pylab.plot_date(x=test_data[self.predict_feat].index, y=test_data[self.predict_feat], fmt="g-")
        pylab.plot(best_result, "r-")
        pylab.plot(val_out, "g-")

        # print len(best_result)
        # print len(val_out)

        pylab.tight_layout()
        # pylab.show()
        pylab.savefig('img/best-' + str(min(mins)) + '=' + str(winner.mse) + '-' + str(self.parents) + self.selection_type + str(self.offsprings) + '.png')
        pylab.clf()

        return min(mins), winner.mse

    def evaluation(self):
        #TODO: change n_samples
        n_samples = self.n_samples
        #random choices make the fitness function in precise
        rows = np.random.choice(range(self.inputs.shape[1]), n_samples)

        #removing duplicates
        rows = list(rows)
        rows = [x for x in rows if not rows.count(x) > 1]
        #
        # samples = self.inputs[rows]

        #seriell
        for g in self.pop:
            g.genotype.update_mse(self.inputs[rows], self.outputs[rows])

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
            if g.genotype.depth() < 60:
                g.mutate()
            else:
                print g.genotype.depth(), ' ',
        pass


def par_wrap(arg):
    sigma, iterations, selection, init_tree_depth, leaf_mutation, node_mutation, n_samples = arg

    print 'start', arg

    e = Evolution(inputs, outputs,
                    iterations = iterations,
                    init_tree_depth=init_tree_depth,
                    selection_type = selection,
                    n_samples = n_samples)
    for genome in e.pop:
        genome.sigma = sigma
        genome.leaf_mutation = leaf_mutation
        genome.node_mutation = node_mutation
    min_, predict_mse = e.run()

    d = {"sigma"           : sigma,
         "iterations"      : iterations,
         "selection"       : selection,
         "init_tree_depth" : init_tree_depth,
         "leaf_mutation"   : leaf_mutation,
         "node_mutation"   : node_mutation,
         "n_samples"       : n_samples,
         'min'             : min_,
         'predict_mse'     : predict_mse,
         'tree'            : e.pop[0].genotype) 
         }

    print 'end', arg
    return d


# sigmas = [0.4, 0.2, 0.1, 0.05, 0.005]#
# iterations = [100]
# selections = ['2+2',  '4+16', '2,2', '4,16']
# init_tree_depths = [3, 6]
# leaf_mutations = [0.1]
# node_mutations = [0.8]
leaf_mutations = [0.1]#, 0.2, 0.1, 0.01]
node_mutations = [0.6]#, 0.6, 0.4, 0.2]
n_samples = [100]

sigmas = [0.2]
iterations = [20]
selections = ['2+2']
init_tree_depths = [12]

args = [sigmas, iterations, selections, init_tree_depths, leaf_mutations, node_mutations, n_samples]

args = list(product(*args))


ds = []

ds = map(par_wrap, args)

# # parallel run
# pool = multiprocessing.Pool(6)
# ds = pool.map(par_wrap, args)
# pool.close()
# pool.join()

keys = ["sigma",
     "iterations",
     "selection",
     "init_tree_depth",
     "leaf_mutation",
     "node_mutation",
     "n_samples",
     'min',
     'predict_mse',
     'tree'
]
#
with open('img/treerun.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, delimiter=';',fieldnames = keys)
        dict_writer.writeheader()
        dict_writer.writerows(ds)

#e = Evolution(train_data, 4, iterations = 10, selection_type = '2+2')
# e.run()
