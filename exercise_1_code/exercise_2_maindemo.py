#!/usr/bin/env python
# Example main

from visualize_results  import *
import fitness_factory
import exercise_2_optimization as optimization
from numpy import arange, linspace, logspace, around, array_str
from itertools import product
from os import rename
import math
from math import pi

def main():
    qApp = QtGui.QApplication(sys.argv)
    logging.basicConfig(level=logging.INFO)

    # Window configuration
    width = 4
    height = 5
    dpi = 100

    # Data file and path definitions
    optimizer_output_path= 'output/optimizer/'
    fitness_statistics = 'fitness_statistics'
    individuals_data = 'individuals'

    # Ranges (both for fitness landscape visualization as well as optimization boundaries
    min_val = -2
    max_val = 2
    grid_size = 0.2
    ranges = arange(min_val, max_val, grid_size)

    # Optimization parameters
    max_iterations = 50
    stepsize = 0.1
    learning_rate = 0.1
    precision = 0.0001

    ds = []


    fitness_function = fitness_factory.SquaredError2D(ranges = ranges)

    crossover_types = ['mean', 'fitnessmean', 'randomweight', 'onepoint', 'onepoint', 'twopoint', 'threepoint']


    optimizer = optimization.genetic(
                                        fitness_function = fitness_function,
                                        precision = precision,
                                        path = optimizer_output_path,
                                        individuals_data = individuals_data,
                                        fitness_statistics = fitness_statistics,
                                        max_iterations = 50,
                                        starting_position = np.array([-1.0]),
                                        pop_size = 10,
                                        select_perc = 0.8,
                                        crossover_type = 'randomweight',
                                        mutation_prob = 0.1)
    _max, _min, _mean = optimizer.run()

    # print _max
    # print _min
    # print _mean
    aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                              fitstat_data=optimizer_output_path + fitness_statistics,
                              fitscape_data=fitness_function.get_fitness_filename(),
                              ranges=ranges,
                              max_iterations=optimizer.get_current_iteration(),
                              width=width, height=height, dpi=dpi,
                              dim=fitness_function.get_dimensionality())
    # aw.export_jpg()
    aw.show()
    sys.exit(qApp.exec_())

if __name__ == "__main__":
    main()
