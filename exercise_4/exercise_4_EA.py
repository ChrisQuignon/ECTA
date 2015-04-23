import numpy as np
import matplotlib.pyplot as plt
#from math import sqrt
from random import randint, shuffle, random

# import multiprocessing
from multiprocessing import Pool

import matplotlib.pyplot as plt
import csv

import fitness as ft


RANGE = ft.dist.shape[0]#TODO fix value

class Genome():
    def __init__(self, mutation_prob, genotype):
        self.mutation_prob = mutation_prob
        self.genotype = []
        self.set_genotype(genotype)

    def set_genotype(self, genotype):
        trackpoints, motor_values = zip(*genotype)

        if len(trackpoints) != len(motor_values):
            print "Error in genotype"
        elif any(t > RANGE or t < 0 for t in trackpoints):
            print 'Error in trackpoints'
        elif any(m < 0.0 or m > 1.0 for m in motor_values):
            print 'error with motor values'
        else:
            self.genotype = genotype



    def mutate(self):
        if random() < self.mutation_prob:

            tp = randint(0, RANGE-1)
            mv = random()

            mutation_idx = randint(0, len(self.genotype)-1)

            mutation = self.genotype
            mutation[mutation_idx] = (tp, mv)
            self.set_genotype(mutation)

    def breed(self, partner):
        basis_tp, basis_mv = zip(*self.genotype)

        partner_tp, partner_mv = zip(*partner.genotype)

        c = randint(0, min(len(partner_tp), len(basis_tp))-1)

        kid_genotype = self.genotype[:c] + partner.genotype[c:]
        # kid_genotype = sorted(kid_genotype, key = lambda x : x[0])#because: why

        return Genome(self.mutation_prob, kid_genotype)

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

p = Genome(0.5, [(0, 1.0)])
fit, _ = ft.vehicle_fitness(p.get_run())
