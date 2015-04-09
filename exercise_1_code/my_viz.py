#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import helpers


class Visualisation():

    def __init__(self):
        self.fig = plt.figure(figsize=plt.figaspect(2.))
        # self.fig, (self.fitness_plot, self.statistics_plot) = plt.subplots(2, sharex=False)
        self.fitness_plot = self.fig.add_subplot(2, 1, 2, projection='3d')
        self.fitness_plot.view_init(270, 90)

        self.statistics_plot = self.fig.add_subplot(2, 1, 1)

        self.data = []
        self.fitscape = []

        self.mins = []
        self.maxs = []
        self.means = []


    def show(self):
        self.fig.show()

    def add_data(self, data):
        x = data[0]
        y = data[1]

        self.data = data
        self.fitness_plot.scatter(x, y, c='r', marker='o', zorder = 4, s=50, linestyle= '-.')

    def add_fitstat(self, fitstat):
        self.mins = fitstat[0]
        self.maxs = fitstat[1]
        self.means = fitstat[2]

        x = np.arange(0., len(fitstat[0]), 1.)

        self.statistics_plot.scatter(x, self.mins, c='r', marker='x', zorder = 4, s=50)
        self.statistics_plot.scatter(x, self.maxs, c='g', marker='x', zorder = 4, s=50)
        self.statistics_plot.scatter(x, self.means, c='b', marker='x', zorder = 4, s=50)

    def add_fitscape(self, fitscape, ranges):
        self.fitscape = fitscape

        if len(fitscape)  == 1:
            self.fitness_plot.plot(ranges, self.fitscape)
        else:
            x, y = np.meshgrid(ranges, ranges)
            self.fitness_plot.plot_surface(x, y, fitscape, rstride=1, cstride=1, cmap=cm.jet,  alpha=0.4, zorder = 10)


    def add_set(self, data, fitstat, fitscape, ranges):
        self.add_data(data)
        self.add_fitstat(fitstat)
        self.add_fitscape(fitscape, ranges)
