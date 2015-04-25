import numpy as np
import matplotlib.pyplot as plt
#from math import sqrt
from random import randint, shuffle, random, gauss

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

        #% MAY BE BAD ( uniform distribution, you know...)

        new_tps = [gauss(mu[0], sigma)%RANGE for mu in self.genotype]

        #the motor command sigma naturally 0.5
        new_mvs = [gauss(x[1], 0.5)%1 for x in self.genotype]#%1
        # print "mutated:"
        # print new_tps, new_mvs
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
            fit =  ft.vehicle_fitness(np.asarray(run))[0]
        except SystemExit:
            return float('inf')
        return fit

class Evolution():
    def __init__(self, iterations, pop_size, selection_type):
        self.iterations = iterations
        self.pop_size = pop_size
        self.selection_type = selection_type

        self.pop = []
        self.sigma = 0.0#does not make sense but works

        self.sigma_increase = 0.0#0.0 means disabled
        self.mutations = 1
        self.improvements = 1

        self.mins = []
        self.maxs = []
        self.means = []

        ##INITIALIZATION
        for _ in range(pop_size):
            # g = Genome([(0, 1.0)])# we start with full power
            g = Genome([(0, 0.0)])# we start with nothing

            self.pop.append(g)

            while g.fitness() < 0:
                steps = randint(0, 10)#%10#TODO scale
                genotype = []
                for _ in range(steps):
                    sp = randint(0, RANGE)
                    mv = random()
                    genotype.append((sp, mv))
                g = Genome(genotype)
                # print genotype

            self.pop.append(g)
        print "initialisation done"


    def run(self):
        for i in range(self.iterations):
            #no more crossover
            self.evaluation()
            self.recombination()
            self.selection()
            # self.mutation()
            # mutation happens inside selection
            # print "select:"
            print map(lambda x: x.fitness(), self.pop)
            # print map(lambda x : x.genotype, self.pop)

        print "DONE"
        print "WINNER"
        self.pop[0].remove_neutral_tps()
        print self.pop[0].genotype
        print "with"
        print self.pop[0].fitness()

        return# self.mins, self.means, self.maxs, self.best_genotype

    def selection(self):
        #environmental selection
        #(1+1), (mu, lambda) and (mu + lambda)
        if '1' in self.selection_type:
            # print 'one on one!'
            parent = self.pop[0]
            kid = copy.deepcopy(self.pop[0])
            kid.mutate(self.sigma)
            self.pop = [parent, kid]

            self.mutations = self.mutations + 1

            #TODO
        elif '+' in self.selection_type:
            print 'plus!'
            #TODO
        elif ',' in self.selection_type.contains:
            print 'comma'
            #TODO
        # print "selection done"

    def recombination(self):

        #TODO 1/5th rule
        if self.sigma_increase != 0.0:
            imp_rate = self.improvements/float(self.mutations)

            if imp_rate > 1/5.0:
                # simga --
                self.sigma = self.sigma - self.sigma_increase
                print "decreased to ", self.sigma
            if imp_rate < 1/5.0:
                #simga ++
                self.sigma = self.sigma + self.sigma_increase
                print "increased to ", self.sigma

                # self.pop[-1].genotype.append(self.pop.genotype[-1])
        else:
            #silently skip
            pass
        pass

    # def mutation(self):
    #     #mutate all the genomes!
    #     for g in self.pop:
    #         g.mutate()

    def evaluation(self):

        #efficient sorting and know whether the min ha schanged

        #we sort the list, but only safe the new keys
        #equivalent to argsort() in numpy
        sort_keys = sorted(range(len(self.pop)), key=lambda k: self.pop[k].fitness())

        #if elemt changed...
        if sort_keys[0] != 0:
            print "improved!"
            self.improvements = 1
            self.mutations = 0

        #we now sort by the index
        self.pop = [self.pop[i] for i in sort_keys]


        #WHEN TO ADD A TRACKPOINT?
        if self.mutations > 10:
            self.pop[0].add_neutral_tp()
            self.improvements = 1
            self.mutations = 0
            print "trackpoint added"



ea =  Evolution(
                iterations = 100,
                pop_size = 1,
                selection_type = '1+1')
ea.run()
