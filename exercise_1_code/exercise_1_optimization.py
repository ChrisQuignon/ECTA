#!/usr/bin/env python
import sys
import os
import time
import csv
import random
import logging
import operator
import shutil
import glob

import math
import numpy as np
from itertools import product

import numpy
import fitness_factory
import helpers

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
        self.fitness_statistics = self.path + fitness_statistics
        helpers.empty_file(self.fitness_statistics)

        self.current_iteration = 0
        self.position = starting_position

        self.old_position = float('nan')
        self.maxfit = float('nan')
        self.minfit = float('nan')
        self.meanfit = float('nan')


        logging.info("Optimization\tinitialized")
        pass

    def step(self):
        pass

    def stop(self):
        #precision is met
        if np.all(np.absolute(self.old_position - self.position)<self.precision):
            return True

        #max iterations are met
        if self.current_iteration >= self.max_iterations:
            return True
        return False

    def export_step(self):

        # print self.position
        # print self.ff.get_point_fitness(self.position)


        #Fitness values
        i = self.current_iteration
        fitness = self.ff.get_point_fitness(self.position)
        self.maxfit = max(fitness, self.maxfit)
        self.minfit = min(fitness, self.minfit)

        if i > 2:
            self.meanfit = self.meanfit * (i-1)/i + fitness * 1 / i
        else:
            self.meanfit = fitness

        # Stack current position and its fitness value
        fit = self.ff.get_point_fitness(self.position)
        data = self.position

        if self.ff.get_dimensionality() > 2:
            data =  self.position.reshape(2, 1)

        data = numpy.vstack((data, fit))

        # Export to file (append)
        helpers.append_file(self.individuals_data, data)

        #Stack fitness statistics
        data = numpy.vstack((self.maxfit, self.meanfit, self.minfit))
        helpers.append_file(self.fitness_statistics, data)

    def run(self):
        while not self.stop():
            self.export_step()
            self.step()
        logging.info("Converged")
        return

    def get_current_iteration(self):
        return self.current_iteration


class HillClimber2DLAB(Optimization):
    def __init__(self, stepsize = 0.1, *args, **kwargs):
        # Construct superclass
        Optimization.__init__(self, *args, **kwargs)
        self.stepsize = stepsize

    def step(self):
        self.current_iteration = self.current_iteration + 1
        self.old_position = self.position

        step = self.stepsize
        dim = self.ff.get_dimensionality()

        best_position = (self.position, float('-Infinity'))

        for q in product([step, 0.0, -step], repeat = dim-1):
            p = q + self.position
            if self.ff.get_point_fitness(p) > self.ff.get_point_fitness(self.position):
                self.position = p
    # def stop(self):
    #     pass

    # def export_step(self):
    #     pass

class SteepestDescent(Optimization):
    def __init__(self, learning_rate = 0.1, inertia = float('nan'), *args, **kwargs):
        # Construct superclass
        Optimization.__init__(self, *args, **kwargs)
        self.learning_rate = learning_rate
        self.inertia = inertia
        self.momentum = 0.0


    def step(self):
        self.current_iteration = self.current_iteration + 1
        self.old_position = self.position

        gradient = self.ff.get_point_gradient(self.position)

        gradient = np.asarray(gradient)
        sd_term = self.learning_rate * gradient

        self.momentum = self.inertia * self.momentum

        delta = sd_term + self.momentum

        self.position = self.position + delta


    # def stop(self):
    #     pass

    # def export_step(self):
    #     pass


class NewtonMethod(Optimization):
    def __init__(self, *args, **kwargs):
        # Construct superclass
        Optimization.__init__(self, *args, **kwargs)


    def step(self):
        self.current_iteration = self.current_iteration + 1
        self.old_position = self.position

        self.position = self.position - self.ff.get_point_gradient(self.position) / self.ff.get_point_second_gradient(self.position)

    # def stop(self):
    #     pass

    # def export_step(self):
    #     pass
