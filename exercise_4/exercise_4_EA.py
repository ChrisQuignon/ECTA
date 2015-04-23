import numpy as np
import matplotlib.pyplot as plt
#from math import sqrt
from random import randint, shuffle, random, gauss

# import multiprocessing
from multiprocessing import Pool

import matplotlib.pyplot as plt
import csv

import fitness as ft


RANGE = ft.dist.shape[0]

class Genome():
    def __init__(self, genotype):
        #self.mutation_prob = 1.0 #TODO: check if this is good
        self.genotype = []
        self.set_genotype(genotype)

    def set_genotype(self, genotype):
        if zip(*genotype):
            trackpoints, motor_values = zip(*genotype)
        else:
            #empty genotype
            self.genotype = [(0.0, 0.0)]
            return

        if len(trackpoints) != len(motor_values):
            print "Error in genotype"
        elif any(t > RANGE or t < 0 for t in trackpoints):
            print 'Error in trackpoints'
        elif any(m < 0.0 or m > 1.0 for m in motor_values):
            print 'error with motor values'
        else:
            self.genotype = genotype



    def mutate(self):
        tps = [x[0] for x in self.genotype]

        if tps[0] != 0:
            tps.insert(0, 0)
        if tps[-1] != RANGE:
            tps.append(RANGE)

        new_tsps = []
        for i in range(1, len(tps)-1):
            #sigma is minium distance to neighbor
            #mu is own position
            mu = tps[i]
            sigma = min([tps[i] - tps[i-1], tps[i+1] - tps[i]])
            new_tsps.append(gauss(mu, sigma))

        new_mvs = [gauss(x[1], 0.5)%1 for x in self.genotype]
        # print new_mvs
        # print new_tsps
        self.set_genotype(zip(new_tsps, new_mvs))

        # #TODO:  Zero mean Gaussian
        # if random() < self.mutation_prob:
        #
        #     tp = randint(0, RANGE-1)
        #     mv = random()
        #
        #     mutation_idx = randint(0, len(self.genotype)-1)
        #
        #     print 'mutating idx' + str(mutation_idx)
        #
        #     mutation = self.genotype
        #     mutation[mutation_idx] = (tp, mv)
    #         self.set_genotype(mutation)

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

    def get_run(self):
        gt_sort = sorted(self.genotype, key = lambda x : x[0])
        run = []
        last_mv = 0.0#TODO: Check whether thats a good idea...
        for i in range(RANGE):
            if len(gt_sort) > 0 and gt_sort[0][0] == i :

                _, last_mv  = gt_sort.pop(0)
            else:
                pass #we just append

            run.append(last_mv)
        return np.asarray(run)

class Evolution():
    def __init__(self, iterations, pop_size, selection_type):
        self.iterations = iterations
        self.pop_size = pop_size
        self.selection_type = selection_type

        self.pop = []

        self.mins = []
        self.maxs = []
        self.means = []

        ##INITIALIZATION
        for _ in range(self.pop_size):
            steps = randint(0, RANGE)/1000#TODO scale
            genotype = []
            for _ in range(steps):
                sp = randint(0, RANGE)
                mv = random()
                genotype.append((sp, mv))
            self.pop.append(Genome(genotype))


    def run(self):
        for i in range(self.iterations):
            #no more crossover
            self.evaluation()
            # self.recombination()
            #self.selection()
            self.mutation()

            # print map(lambda x : x.genotype, self.pop)
            print map(lambda x: ft.vehicle_fitness(x.get_run())[0], self.pop)

        return# self.mins, self.means, self.maxs, self.best_genotype

    def selection(self):
        #environmental selection
        #ELIF selction_type
        #(1+1), (mu, lambda) and (mu + lambda)

        pass

    def recombination(self):
        #TODO 1/5th rule

        #improvement/mutation > 1/5.0: simga --
        #improvement/mutation < 1/5.0: simga ++
        pass

    def mutation(self):
        #zero mean gaussian mutation on all
        # print 'mutate'
        #mutate all the genomes!
        for g in self.pop:
            g.mutate()

    def evaluation(self):
        self.pop = sorted(self.pop, key = lambda g : ft.vehicle_fitness(g.get_run())[0])
        pass



ea =  Evolution(
                iterations = 1,
                pop_size = 4,
                selection_type = '+')
ea.run()
