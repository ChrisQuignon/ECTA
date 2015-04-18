#!/usr/bin/env python
import sys
import os
import time
import csv
# import random
import logging
import operator
import shutil
import glob

import numpy
import fitness_factory
import helpers


import numpy as np
from itertools import product
from random import random, triangular, choice

class Optimization():
    def __init__(self, fitness_function, precision, path, individuals_data, fitness_statistics, max_iterations, starting_position = 0.0):
        self.ff = fitness_function
        self.precision = precision
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.path = path
        self.gen = 0
        files = glob.glob(self.path + str('*'))
        for f in files:
            os.remove(f)
        self.individuals_data = self.path + individuals_data
        helpers.empty_file(self.individuals_data)
        # self.fitness_statistics = self.path + fitness_statistics
        # helpers.empty_file(self.fitness_statistics)

        self.current_iteration = 0
        self.position = starting_position

        self.old_position = np.array([float('nan')])
        self.maxfit = float('nan')
        self.minfit = float('nan')
        self.meanfit = float('nan')

        logging.info("Optimization\tinitialized")

    # def step(self):

    def stop(self):
        #precision is met
        # if np.all(np.absolute(self.old_position - self.position)<self.precision):
        #     print 'precision'
        #     return True

        #max iterations are met
        if self.current_iteration >= self.max_iterations:
            return True

        #position check
        if np.any(self.position > 10**10):
            logging.info("Diverged. Last position: %f", self.position)
            return True

        if np.any(self.position < -10**10):
            logging.info("Diverged. Last position: %f", self.position)
            return True
        return False

    def export_step(self):
        #Fitness values
        i = self.current_iteration
        fitness = self.ff.get_point_fitness(self.position)
        self.maxfit = max(fitness, self.maxfit)
        self.minfit = min(fitness, self.minfit)

        if i > 2:
            self.meanfit = self.meanfit * (i-1)/i + fitness * 1 / i
        else:
            self.meanfit = fitness

        # # Stack current position and its fitness value
        fit = self.ff.get_point_fitness(self.position)
        data = self.position

        #weird stuff is weird...
        if self.ff.get_dimensionality() == 3:
            data = numpy.append(data, fit)
        if self.ff.get_dimensionality() == 2:
            data = numpy.vstack((data, fit))

        # # Export to file (append)
        helpers.append_file(self.individuals_data, data)
        #
        # #Stack fitness statistics
        # data = numpy.vstack((self.maxfit, self.meanfit, self.minfit))
        # helpers.append_file(self.fitness_statistics, data)

    def run(self):
        while not self.stop():
            self.export_step()
            self.step()
        logging.info("Converged")
        # print self.maxfit
        # print self.minfit
        # print self.meanfit
        return (self.maxfit, self.minfit, self.meanfit)

    def get_current_iteration(self):
        return self.current_iteration


class Genetic(Optimization):
        def __init__(self, pop_size, select_perc, crossover_type, mutation_prob, *args, **kwargs):
            # Construct superclass
            Optimization.__init__(self, *args, **kwargs)

            self.pop_size = pop_size
            self.select_perc = select_perc
            self.crossover_type = crossover_type
            self.mutation_prob = mutation_prob
            # Two kinds of representation: real valued, bit string
            # Population size (compare 3 values)
            # Selection (compare 3 values)
            # Crossover (compare 3 values)
            # Mutation (compare 3 values)

            self.pop = np.empty( shape=(0, 0) )

            self.initialize()

        def initialize(self):
            for i in range(self.pop_size):
                _min = self.ff.ranges[0]
                _max = self.ff.ranges[-1]
                individual = self.position

                s = individual.shape
                individual.flatten()
                for i in range(individual.size):
                    individual[i] = random() *(_max-_min)+_min
                individual.reshape(s)

                if self.pop.size == 0:
                    self.pop = np.asarray(individual)
                else:
                    self.pop = np.vstack((self.pop, individual))

        def evaluation(self):
            self.pop = sorted(self.pop, key=(lambda p :self.ff.get_point_fitness(p)))
            self.pop.reverse()
            self.position = self.pop[0]


        def selection(self):
            n = int(len(self.pop) * self.select_perc)
            self.pop = self.pop[:n]

        def crossover(self):

            #elitism
            newpop = [self.pop[0]]

            if self.crossover_type == 'mean':
                for _ in range(self.pop_size-1):
                    parent_a = choice(self.pop)
                    parent_b = choice(self.pop)

                    offspring = (parent_a + parent_b)/2.0
                    newpop.append(offspring)

            elif self.crossover_type == 'fitnessmean':
                for _ in range(self.pop_size-1):
                    parent_a = choice(self.pop)
                    parent_b = choice(self.pop)

                    f_a = self.ff.get_point_fitness(parent_a)
                    f_b = self.ff.get_point_fitness(parent_b)

                    offspring = (parent_a * f_a + parent_b * f_a) / (f_a + f_b)
                    newpop.append(offspring)

            elif self.crossover_type == 'randomweight':
                for _ in range(self.pop_size-1):
                    parent_a = choice(self.pop)
                    parent_b = choice(self.pop)

                    f_a = random()
                    f_b = random()

                    offspring = (parent_a * f_a + parent_b * f_a) / (f_a + f_b)
                    newpop.append(offspring)


            elif self.crossover_type == 'onepoint':

                for _ in range(self.pop_size-1):

                    parent_a = choice(self.pop)
                    parent_b = choice(self.pop)

                    # print parent_a
                    # print parent_b

                    offspring = []

                    for i in range(parent_a.size):
                        a_front, a_tail = str(parent_a[i]).split('.')
                        b_front, b_tail = str(parent_b[i]).split('.')

                        try:
                            offspring.append(float(str(a_front + '.' + b_tail)))
                        except ValueError:
                            print "Not a float: " + str(a_front + '.' + b_tail)
                            return #pech gehabt...

                    newpop.append(np.asarray(offspring))

            elif self.crossover_type == 'twopoint':
                for _ in range(self.pop_size-1):
                    parent_a = choice(self.pop)
                    parent_b = choice(self.pop)

                    # print parent_a
                    # print parent_b

                    offspring = []

                    for i in range(parent_a.size):
                        a_front, a_tail = str(parent_a[i]).split('.')
                        b_front, b_tail = str(parent_b[i]).split('.')

                        for i in range(1, 2):
                            temp = a_tail[i:]
                            a_tail = a_tail[:i] + b_tail[i:]
                            b_tail = b_tail[:i] + temp

                        try:
                            offspring.append(float(str(a_front + '.' + b_tail)))
                        except ValueError:
                            print "Not a float: " + str(a_front + '.' + b_tail)
                            return #pech gehabt...

                    newpop.append(np.asarray(offspring))

            elif self.crossover_type == 'threepoint':
                for _ in range(self.pop_size-1):
                    parent_a = choice(self.pop)
                    parent_b = choice(self.pop)

                    offspring = []

                    # print parent_a
                    # print parent_b

                    for i in range(parent_a.size):
                        a_front, a_tail = str(parent_a[i]).split('.')
                        b_front, b_tail = str(parent_b[i]).split('.')

                        for i in range(1, 3):
                            temp = a_tail[i:]
                            a_tail = a_tail[:i] + b_tail[i:]
                            b_tail = b_tail[:i] + temp

                        try:
                            offspring.append(float(str(a_front + '.' + b_tail)))
                        except ValueError:
                            print "Not a float: " + str(a_front + '.' + b_tail)
                            return #pech gehabt...


                    newpop.append(np.asarray(offspring))

            else:
                print 'no crossover for type ' +  self.crossover_type

            self.pop = newpop


        def mutation(self):
            t = self.crossover_type
            if t == 'onepoint' or t == 'twopoint' or t == 'threepoint':
                for i in self.pop:
                    r = random()
                    if r < self.mutation_prob:
                        s =  str(i)
                        idx = int(random() * len(s))

                        while not s[idx].isdigit():
                            idx = int(random() * len(s))
                        s = s[:idx] + choice('0123456789') +s[idx+1:]

                        # print 'mutating ', i , ' -> ', s
                        i = np.asarray(s)
            else:
                for i in self.pop:
                    r = random()
                    if r < self.mutation_prob:

                        r = random() -0.5
                        s = i +r
                        # print 'mutating ', i , ' -> ', s
                        i = s


        def step(self):
            # print self.position
            self.current_iteration = self.current_iteration + 1
            # print self.position
            # self.pop = sorted(self.pop, key=(lambda p :self.ff.get_point_fitness(p)))
            # print self.pop



            dim = self.ff.get_dimensionality()

            self.evaluation()
            self.selection()
            self.crossover()
            self.mutation()

            # print self.pop

            # self.current_iteration = self.max_iterations
            # print self.current_iteration
            # best_position = (self.position, float('-Infinity'))
