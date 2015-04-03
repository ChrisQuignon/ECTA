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

import numpy
import fitness_factory
import helpers

class Optimization():
    def __init__(self, fitness_function, precision, path, individuals_data, fitness_statistics, max_iterations):
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


        logging.info("Optimization\tinitialized")
        pass

    def step(self):
        pass

    def stop(self):
        pass

    def export_step(self):
        pass

    def run(self):
        while not self.stop():
            self.export_step()
            self.step()
        logging.info("Converged")
        return

    def get_current_iteration(self):
        return self.current_iteration


class HillClimber2DLAB(Optimization):
    def __init__(self, starting_position = 0.0, stepsize = 0.1, *args, **kwargs):
        # Construct superclass
        Optimization.__init__(self, *args, **kwargs)
        self.stepsize = stepsize
        self.current_iteration = 0
        self.position = starting_position

        self.maxfit = float('nan')
        self.meanfit = 0.0

    def step(self):
        self.current_iteration = self.current_iteration + 1

        left = self.ff.get_point_fitness(self.position - self.stepsize)
        middle = self.ff.get_point_fitness(self.position)
        right = self.ff.get_point_fitness(self.position + self.stepsize)

        if left >= middle and left >= right:
            self.position = self.position - self.stepsize
        elif middle <= right and left <= right:
            self.position = self.position + self.stepsize


        #Fitness values
        i = self.current_iteration
        fitness = self.ff.get_point_fitness(self.position)
        self.maxfit = max(fitness, self.maxfit)

        if i > 2:
            self.meanfit = self.meanfit * (i-1)/i + fitness * 1 / i
        else:
            self.meanfit = fitness

        #TODO: check for higher dimensionality
        #self.ff.get_dimensionality

    def stop(self):
        if self.current_iteration >= self.max_iterations:
            return True
        return False

    def export_step(self):
        # Stack current position and its fitness value
        data = numpy.vstack((self.position, self.ff.get_point_fitness(self.position)))
        # Export to file (append)
        helpers.append_file(self.individuals_data, data)

        # # Stack fitness statistics
        data = numpy.vstack((self.maxfit, self.meanfit))
        helpers.append_file(self.fitness_statistics, data)

class steepestDescent(Optimization):
    def __init__(self, starting_position = 0.0, learning_rate = 0.1, inertia = float('nan'), *args, **kwargs):
        # Construct superclass
        Optimization.__init__(self, *args, **kwargs)
        self.current_iteration = 0
        self.position = starting_position
        self.learning_rate = learning_rate
        self.inertia = inertia
        self.momentum = 0.0


    def step(self):
        self.current_iteration = self.current_iteration + 1

        delta = self.ff.get_point_fitness(self.position)
        steepest_descent = -1.0 * self.learning_rate * delta

        #no inertia means no momentum
        if not math.isnan(self.inertia):
            self.momentum = self.inertia * self.momentum

        self.position = steepest_descent + self.momentum

        #TODO: check for higher dimensionality
        #self.ff.get_dimensionality

    def stop(self):
        if self.current_iteration >= self.max_iterations:
            return True
        return False

    def export_step(self):
        # Stack current position and its fitness value
        data = numpy.vstack((self.position, self.ff.get_point_fitness(self.position)))
        # Export to file (append)
        helpers.append_file(self.individuals_data, data)


class newtonMethod(Optimization):
    def __init__(self, starting_position = 0.0, *args, **kwargs):
        # Construct superclass
        Optimization.__init__(self, *args, **kwargs)
        self.current_iteration = 0
        self.position = starting_position


    def step(self):
        self.current_iteration = self.current_iteration + 1
        self.position = self.position - self.ff.get_point_gradient(self.position) / self.ff.get_point_second_gradient(self.position)

    def stop(self):
        if self.current_iteration >= self.max_iterations:
            return True
        return False

    def export_step(self):
        # Stack current position and its fitness value
        data = numpy.vstack((self.position, self.ff.get_point_fitness(self.position)))
        # Export to file (append)
        helpers.append_file(self.individuals_data, data)
