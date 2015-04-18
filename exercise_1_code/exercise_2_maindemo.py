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
    precision = 0.0001
    ranges = arange(min_val, max_val, grid_size)

    # Optimization parameters
    max_iterations = 50

    ds = []

    pop_sizes = [10, 100, 1000]
    select_percs = [0.1, 0.5, 1]
    mutation_probs = [0.01, 0.1, 0.3]


    fitness_function = fitness_factory.Trimodal2D(ranges = ranges)

    #THREE REAL VALUES, THREE ARITHMETIC
    crossover_types = ['mean', 'fitnessmean', 'randomweight', 'onepoint', 'twopoint', 'threepoint']
    # ffs = [fitness_factory.Plateau3D, fitness_factory.SquaredError2D, fitness_factory.Trimodal2D]

    # fitness_function = fitness_factory.Plateau3D(ranges = ranges)
    #
    # optimizer = optimization.genetic(
    #                                     fitness_function = fitness_function,
    #                                     precision = precision,
    #                                     path = optimizer_output_path,
    #                                     individuals_data = individuals_data,
    #                                     fitness_statistics = fitness_statistics,
    #                                     max_iterations = 100,
    #                                     starting_position = np.array([-1.0, 1.0]),
    #                                     # starting_position = np.array([1.0]),
    #                                     pop_size = 10,
    #                                     select_perc = 0.8,
    #                                     crossover_type = 'threepoint',
    #                                     mutation_prob = 0.1)
    # _max, _min, _mean = optimizer.run()
    #
    # print fitness_function.__class__.__name__
    # # print crossover_types[i]
    # print _max
    # print _min
    # print _mean

    #FULL ANALYSIS
    ds = []
    def full_analysis():

        # Ranges (both for fitness landscape visualization as well as optimization boundaries
        min_val = -2
        max_val = 2
        grid_size = 0.2
        precision = 0.0001
        ranges = arange(min_val, max_val, grid_size)

        # Optimization parameters
        max_iterations = 30

        pop_sizes = [10, 50, 100]# 100 ]#, 1000]
        select_percs = [0.1, 0.5, 1]
        mutation_probs = [0.01, 0.1, 0.3]


        fitness_function = fitness_factory.Trimodal2D(ranges = ranges)

        #THREE REAL VALUES, THREE ARITHMETIC
        crossover_types = ['onepoint', 'twopoint', 'threepoint', 'mean', 'fitnessmean', 'randomweight']
        ffs = [fitness_factory.Plateau3D, fitness_factory.SquaredError2D, fitness_factory.Trimodal2D]

        for pop_size in pop_sizes:
            for select_perc in select_percs:
                for mutation_prob in mutation_probs:
                    for i in range(len(crossover_types)):

                        if i < 3:
                            sp = np.array([-1.0, 1.0])#used as a schema
                        else:
                            sp = np.array([-1.0])#used as a schema

                        for fitness_function_name in ffs:

                            ranges = arange(min_val, max_val, grid_size)

                            if fitness_function_name.__name__ == "SquaredError2D":
                                min_val = -2
                                max_val = 2
                                grid_size = 0.2
                                ranges =  arange(min_val, max_val, grid_size)
                                starting_position = np.array([-1.0])

                            if fitness_function_name.__name__ == "Trimodal2D":
                                min_val = -3.4
                                max_val = -0.6
                                grid_size = 0.2
                                ranges =  arange(min_val, max_val, grid_size)
                                starting_position = np.array([-1.0])

                            if fitness_function_name.__name__ == "Plateau3D":
                                min_val = -2
                                max_val = 2
                                grid_size = 0.5
                                ranges =  arange(min_val, max_val, grid_size)
                                starting_position = np.array([-1.0, 1.0])


                            fitness_function = fitness_function_name(ranges = ranges)

                            optimizer = optimization.Genetic(
                                                                fitness_function = fitness_function,
                                                                precision = precision,
                                                                path = optimizer_output_path,
                                                                individuals_data = individuals_data,
                                                                fitness_statistics = fitness_statistics,
                                                                max_iterations = 100,
                                                                starting_position = starting_position,
                                                                pop_size = pop_size,
                                                                select_perc = select_perc,
                                                                crossover_type = crossover_types[i],
                                                                mutation_prob = mutation_prob)
                            _max, _min, _mean = optimizer.run()

                            aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                                                      fitstat_data=optimizer_output_path + fitness_statistics,
                                                      fitscape_data=fitness_function.get_fitness_filename(),
                                                      ranges=ranges,
                                                      max_iterations=optimizer.get_current_iteration(),
                                                      width=width, height=height, dpi=dpi,
                                                      dim=fitness_function.get_dimensionality())
                            aw.export_jpg()

                            fname = optimizer.__class__.__name__ + "/" + fitness_function_name.__name__
                            fname = fname +  "_" + str(pop_size)
                            fname = fname +  "_" + str(select_perc)
                            fname = fname +  "_" + str(mutation_prob)
                            fname = fname +  "_" + crossover_types[i]
                            print fname

                            rename("export.jpg", "output/images/" + fname)


                            ds.append({
                                        'min':float(_min),
                                        'mean':float(_mean),
                                        'max':float(_max),
                                        'ff':fitness_function_name.__name__,
                                        'opt':optimizer.__class__.__name__,
                                        'max_iterations':max_iterations,
                                        'pop_size':pop_size,
                                        'select_perc':select_perc,
                                        'crossover_type':crossover_types[i],
                                        'mutation_prob':mutation_prob
                                        })


    full_analysis()

    # print ds

    #Export Data into data.csv
    keys = ['min',
            'mean',
            'max',
            'ff',
            'opt',
            'max_iterations',
            'pop_size',
            'select_perc',
            'crossover_type',
            'mutation_prob']
    with open('data_genetic.csv', 'wb') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(ds)

    # aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
    #                           fitstat_data=optimizer_output_path + fitness_statistics,
    #                           fitscape_data=fitness_function.get_fitness_filename(),
    #                           ranges=ranges,
    #                           max_iterations=optimizer.get_current_iteration(),
    #                           width=width, height=height, dpi=dpi,
    #                           dim=fitness_function.get_dimensionality())
    # # aw.export_jpg()
    # aw.show()
    # sys.exit(qApp.exec_())

if __name__ == "__main__":
    main()
