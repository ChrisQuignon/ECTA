import numpy as np
import matplotlib.pyplot as plt
#from math import sqrt
from random import randint, shuffle, random, gauss, choice
from itertools import groupby

# import multiprocessing
from multiprocessing import Pool

import matplotlib.pyplot as plt
import csv

import vehicle_simulation as ft

import copy


RANGE = ft.dist.shape[0]

class Genome():
    def __init__(self, genotype):
        self.genotype = []
        self.set_genotype(genotype)

    def set_genotype(self, genotype):

        #remove duplicate position
        gt = genotype
        gt.sort()
        genotype = []

        last_pos = -1
        for pos, mv in gt:
            if last_pos == pos:
                pass
            else:
                genotype.append((pos, mv))
            last_pos = pos
        # genotype = list(k for k,_ in groupby(gt))

        if zip(*genotype):
            trackpoints, motor_values = zip(*genotype)
        else:
            #empty genotype
            print "empty genome"
            self.genotype = [(0.0, 1.0)]
            return

        if len(trackpoints) != len(motor_values):
            print "Error in genotype"
        elif any(t > RANGE or t < 0 for t in trackpoints):
            print 'Error in trackpoints: ', trackpoints
        elif any(m < 0.0 or m > 1.0 for m in motor_values):
            print 'error with motor values: ', motor_values
        else:
            self.genotype = genotype


    def remove_neutral_tps(self):

        last_mv = self.genotype[0][1]

        del_idxs = []

        for i in range(1, len(self.genotype)):
            if self.genotype[i][1] == last_mv:
                del_idxs.append(i)
            last_mv = self.genotype[i][1]

        #remove multiple idx from list
        self.genotype = [elem for idx, elem in enumerate(self.genotype) if idx not in del_idxs]

    def add_neutral_tp(self):
        self.remove_neutral_tps()

        pos = int(random()*RANGE)

        last_mv = self.genotype[0][1]

        for idx, (tp, mv) in enumerate(self.genotype):
            if pos < tp:
                self.genotype.insert(idx, (pos, last_mv))
                return
            else:
                last_mv = mv
        else:#pos > max(tp)
            self.genotype.append((pos, last_mv))


    def mutate(self, sigma):

        #exponential grothdouble the number of trackpoints
        for _ in range(len(self.genotype)):
            self.add_neutral_tp()

        #modulo (%) MAY BE BAD ( uniform distribution, you know...)
        #We mutate everything
        new_tps = [(gauss(pos[0], sigma)*RANGE/2.0)%RANGE for pos in self.genotype]


        new_mvs = [gauss(pos[1], sigma/2.0)%1 for pos in self.genotype]#%1

        new_tps = map(int, new_tps)
        # new_tps = map (lambda a : a%RANGE, new_tps)
        #
        # print 'new_tps:'
        # print new_tps
        #
        # print 'new_mvs:'
        # print new_mvs
        #
        # print 'zip:'
        # print zip(new_tps, new_mvs)
        #
        # print ''
        # print mutate
        self.set_genotype(zip(new_tps, new_mvs))


    # def breed(self, partner):
    #     basis_tp, basis_mv = zip(*self.genotype)
    #
    #     partner_tp, partner_mv = zip(*partner.genotype)
    #
    #     c = randint(0, min(len(partner_tp), len(basis_tp))-1)
    #
    #     kid_genotype = self.genotype[:c] + partner.genotype[c:]
    #     # kid_genotype = sorted(kid_genotype, key = lambda x : x[0])#because: why
    #
    #     return Genome(self.mutation_prob, kid_genotype)

    def fitness(self):
        gt_sort = sorted(self.genotype, key = lambda x : x[0])
        run = []
        last_mv = 0.0#TODO: Check whether thats a good idea...
        for i in range(RANGE):
            if len(gt_sort) > 0 and gt_sort[0][0] == i :

                _, last_mv  = gt_sort.pop(0)
            else:
                pass #we just append
            run.append(last_mv)
        try:
            fit =  ft.vehicle_fitness(np.asarray(run))#[0]
        except SystemExit:
            return float('inf')
        return fit

class Evolution():
    def __init__(self, iterations, selection_type):
        self.iterations = iterations

        self.selection_type = [c for c in selection_type if not c.isdigit()][0]
        self.parents, self.offsprings = map(int, selection_type.split(self.selection_type))

        self.pop = [Genome([(0, 1.0)])]
        self.sigma = 0.002 #does not make sense but works

        self.sigma_increase = 0.1#0.0 means disabled - it just drops to zero anyway
        self.mutations = 1
        self.improvements = 1

        self.mins = []
        self.maxs = []
        self.means = []

        ##INITIALIZATION
        for _ in range(1, self.parents):
            self.pop.append(Genome([(0, 1.0)]))

        # print "initialisation done"


    def run(self):
        for i in range(self.iterations):
            #no more crossover
            self.evaluation()
            self.recombination()
            self.selection()
            # self.mutation()
            # mutation happens inside selection

            print round(self.improvements/float(self.mutations), 2), 'succ with :', self.sigma
            # print round(self.improvements/float(self.mutations), 2)
            # print map(lambda x: x.fitness(), self.pop)
            # print map(lambda x : x.genotype, self.pop)


        self.pop[0].remove_neutral_tps()
        print self.parents, self.selection_type, self.offsprings, 'terminated after ', self.iterations, " iterations:"
        print 'fitness: '
        print self.pop[0].fitness()
        print "track:"
        print self.pop[0].genotype

        return# self.mins, self.means, self.maxs, self.best_genotype

    def selection(self):

        parents = [self.pop[i] for i in range(self.parents)]
        kids = []

        for _ in range(self.offsprings):
            parent = choice(parents)
            kid = copy.deepcopy(self.pop[0])
            kid.mutate(self.sigma)
            self.mutations = self.mutations + 1

            #we don't care about our kids
            # while kid.fitness() == float('inf'):
            #     kid.mutate(self.sigma)
            #     print "dead kid"
            kids.append(kid)

        if self.selection_type == '+':
            self.pop = parents + kids

        elif self.selection_type == ',':
            self.pop = kids
        else:
            print "selection type unkown!"

    def recombination(self):

        #TODO 1/5th rule
        if self.sigma_increase != 0.0:
            imp_rate = self.improvements/float(self.mutations)

            if imp_rate > 1/5.0:
                # simga --
                self.sigma = self.sigma * (1 + self.sigma_increase)
                # print "increased to ", self.sigma
            if imp_rate < 1/5.0:
                #simga ++
                self.sigma = self.sigma * (1 - self.sigma_increase)
                # print "decreased to ", self.sigma
        else:
            #silently skip
            pass
        pass

    def evaluation(self):

        #efficient sorting and know whether the min ha schanged

        #we sort the list, but only safe the new keys
        #equivalent to argsort() in numpy
        sort_keys = sorted(range(len(self.pop)), key=lambda k: self.pop[k].fitness())

        #if elemt changed...
        if sort_keys[0] != 0:
            print "improved!"
        #     self.improvements = self.improvements +1

        #we now sort by the index
        self.pop = [self.pop[i] for i in sort_keys]

        self.improvements = 0;
        for i in self.pop:
            if i.fitness() != float("inf"):
                self.improvements = self.improvements +1

        self.mutations = len(self.pop)




ea =  Evolution(
                iterations = 50,
                selection_type = '2+20')
ea.run()
