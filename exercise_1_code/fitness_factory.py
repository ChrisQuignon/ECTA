#!/usr/bin/env python
import sys
import os
import time
import csv
import operator
import functools
import logging

import numpy as np

import helpers

class FitnessLandscape():
    def __init__(self, output_file, ranges):
        self.ranges = ranges
        if len(self.ranges) is 0:
            logging.error("No ranges set. Your FitnessLandscape is useless now")
        self.output_file = "output/fitness_functions/fitness_" + str(output_file)
        self.fitgraph = self.produce_fitness()
        self.export()
        pass

    def export(self):
        helpers.write_file(self.output_file, self.fitgraph)
        pass

    def return_fitness(self):
        if len(self.fitgraph) is 0:
            logging.error("Current fitness graph is empty")
        return self.fitgraph

    def produce_fitness(self):
        pass

    def get_dimensionality(self):
        pass

    def get_point_fitness(self, point):
        pass

    def get_point_gradient(self, point):
        pass

    def get_ranges(self):
        return self.ranges

    def get_fitness_filename(self):
        return self.output_file

    def get_fitness_name(self):
        pass

class Plateau3D(FitnessLandscape):
    def __init__(self, *args, **kwargs):
        FitnessLandscape.__init__(self, output_file='plateau', *args,**kwargs)

    def produce_fitness(self):
        x = self.ranges
        y = self.ranges
        x, y = np.meshgrid(x, y)
        self.fit = 4*(1-x)**2*np.exp(-(x**2)-(y+1)**2) -15*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2) -(1/3)*np.exp(-(x+1)**2 - y**2)-1*(2*(x-3)**7 -0.3*(y-4)**5+(y-3)**9)*np.exp(-(x-3)**2-(y-3)**2);
        return self.fit

    def get_dimensionality(self):
        return 3

    def get_point_fitness(self, point):
        self.fit = 4*(1-point[0])**2*np.exp(-(point[0]**2)-(point[1]+1)**2) -15*(point[0]/5 - point[0]**3 - point[1]**5)*np.exp(-point[0]**2-point[1]**2) -(1/3)*np.exp(-(point[0]+1)**2 - point[1]**2)-1*(2*(point[0]-3)**7 -0.3*(point[1]-4)**5+(point[1]-3)**9)*np.exp(-(point[0]-3)**2-(point[1]-3)**2)
        return self.fit

    def get_point_gradient(self, point):
        dx =-8*np.exp(-(point[0]**2)-(point[1]+1)**2)*((1-point[0])+point[0]*(1-point[0])**2)-15*np.exp(-point[0]**2-point[1]**2)*((0.2-3*point[0]**2) -2*point[0]*(point[0]/5 - point[0]**3 - point[1]**5)) +(2/3)*(point[0]+1)*np.exp(-(point[0]+1)**2 - point[1]**2)-1*np.exp(-(point[0]-3)**2-(point[1]-3)**2)*(14*(point[0]-3)**6-2*(point[0]-3)*(2*(point[0]-3)**7-0.3*(point[1]-4)**5+(point[1]-3)**9));
        dy =-8*(point[1]+1)*(1-point[0])**2*np.exp(-(point[0]**2)-(point[1]+1)**2) -15*np.exp(-point[0]**2-point[1]**2)*(-5*point[1]**4 -2*point[1]*(point[0]/5 - point[0]**3 - point[1]**5)) +(2/3)*point[1]*np.exp(-(point[0]+1)**2 - point[1]**2)-1*np.exp(-(point[0]-3)**2-(point[1]-3)**2)*((-1.5*(point[1]-4)**4+9*(point[1]-3)**8)-2*(point[1]-3)*(2*(point[0]-3)**7-0.3*(point[1]-4)**5+(point[1]-3)**9));
        return (dx, dy)

    def get_fitness_name(self):
        return "4*(1-x)^2*exp(-(x^2)-(y+1)^2) -15*(x/5 - x^3 - y^5)*exp(-x^2-y^2) -(1/3)*exp(-(x+1)^2 - y^2)-1*(2*(x-3)^7 -0.3*(y-4)^5+(y-3)^9)*exp(-(x-3)^2-(y-3)^2)"

class Bimodal2D(FitnessLandscape):
    def __init__(self, *args, **kwargs):
        FitnessLandscape.__init__(self, output_file='bimodal', *args,**kwargs)

    def produce_fitness(self):
        self.x = self.ranges
        self.fit = -1 * (self.x-1)**2 * (self.x+1)**2
        return (self.fit)

    def get_dimensionality(self):
        return 2

    def get_point_fitness(self, point):
        return -1 * ((point-1)**2 * (point+1)**2)

    def get_point_gradient(self, point):
        return - (2*point - 2)*(point + 1)**2 - (2*point + 2)*(point - 1)**2

    def get_point_second_gradient(self, point):
        return - 2*(point - 1)**2 - 2*(point + 1)**2 - 2*(2*point - 2)*(2*point + 2)

class Multimodal3D(FitnessLandscape):
    def __init__(self, *args, **kwargs):
        FitnessLandscape.__init__(self, output_file='multimodal3D', *args,**kwargs)

    def produce_fitness(self):
        self.x = self.ranges
        self.y = self.ranges
        self.x, self.y = np.meshgrid(self.x, self.y)
        self.fit = -1.0 * (  (np.cos(self.x))**2  * (np.sin(self.y))**2 - 1.0)**2 + 2.
        return (self.fit)

    def get_dimensionality(self):
        return 3

    def get_point_fitness(self, point):
        self.fit = -1.0 * (  (np.cos(point[0]))**2  * (np.sin(point[1]))**2 - 1.0)**2 + 2.
        return self.fit

    def get_point_gradient(self, point):
        dx = 2*np.sin(point[0])*np.sin(point[1])*(np.cos(point[0])*np.sin(point[1]) - 1)*(np.cos(point[0])*np.sin(point[1]) + 1)**2 + 2*np.sin(point[0])*np.sin(point[1])*(np.cos(point[0])*np.sin(point[1]) - 1)**2*(np.cos(point[0])*np.sin(point[1]) + 1)
        dy = - 2*np.cos(point[0])*np.cos(point[1])*(np.cos(point[0])*np.sin(point[1]) - 1)*(np.cos(point[0])*np.sin(point[1]) + 1)**2 - 2*np.cos(point[0])*np.cos(point[1])*(np.cos(point[0])*np.sin(point[1]) - 1)**2*(np.cos(point[0])*np.sin(point[1]) + 1)
        return (dx, dy)

class Trimodal2D(FitnessLandscape):
    def __init__(self, *args, **kwargs):
        FitnessLandscape.__init__(self, output_file='trimodal', *args,**kwargs)

    def produce_fitness(self):
        self.x = self.ranges
        self.fit = -1 * (self.x+1)**2 * (self.x+2)**2 * (self.x+3)**2 - 0.1 * (self.x+2)**2 + 2
        return (self.fit)

    def get_dimensionality(self):
        return 2

    def get_point_fitness(self, point):
        point_fit = -1 * (point+1)**2 * (point+2)**2 * (point+3)**2 - 0.1 * (point+2)**2 + 2
        return point_fit

    def get_point_gradient(self, point):
        point_gradient = - point/5 - (2*point + 2)*(point + 2)**2*(point + 3)**2 - (2*point + 4)*(point + 1)**2*(point + 3)**2 - (2*point + 6)*(point + 1)**2*(point + 2)**2 - 2./5.
        return point_gradient

    def get_point_second_gradient(self, point):
        point_second_gradient = - 2*(point + 1)**2*(point + 2)**2 - 2*(point + 1)**2*(point + 3)**2 - 2*(point + 2)**2*(point + 3)**2 - 2*(2*point + 2)*(2*point + 4)*(point + 3)**2 - 2*(2*point + 2)*(2*point + 6)*(point + 2)**2 - 2*(2*point + 4)*(2*point + 6)*(point + 1)**2 - 1./5.
        return point_second_gradient

class SquaredError2D(FitnessLandscape):
    def __init__(self, *args, **kwargs):
        FitnessLandscape.__init__(self, output_file='squared_error', *args,**kwargs)

    def produce_fitness(self):
        self.fit = -(self.ranges)**2
        return (self.fit)

    def get_dimensionality(self):
        return 2

    def get_point_fitness(self, point):
        return -((point)**2)

    def get_point_gradient(self, point):
        return -(2*(point))

    def get_point_second_gradient(self, point):
        return -2

class Triangle2D(FitnessLandscape):
    def __init__(self, *args, **kwargs):
        FitnessLandscape.__init__(self, output_file='triangle', *args,**kwargs)

    def produce_fitness(self):
        self.fit = -np.abs(self.ranges)
        return (self.fit)

    def get_dimensionality(self):
        return 2

    def get_point_fitness(self, point):
        return -abs(point)

    def get_point_gradient(self, point):
        if point < 0:
            return 1
        elif point > 0:
            return -1
        else:
            return 0

    def get_point_second_gradient(self, point):
        if point < 0 or point > 0:
            return 0
        else:
            return float('NaN')
