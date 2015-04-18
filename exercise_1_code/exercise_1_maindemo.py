#!/usr/bin/env python
# Example main

from visualize_results  import *
import fitness_factory
import exercise_1_optimization as optimization
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


    def latest_analysis():
        # test_vals = [0.0, 0.001, 0.01, 0.01, 0.1, 0.2, 0.4, 0.8, 1.0, 1.2]
        test_vals = np.hstack(([0], logspace(0, 2, 9)/250))
        # test_vals = [0.0]

        for fitness_function_name in [fitness_factory.Plateau3D, fitness_factory.SquaredError2D, fitness_factory.Trimodal2D]:#fitness_factory.SquaredError2D, fitness_factory.Trimodal2D,
            for optimizer_name in [optimization.HillClimber, optimization.SteepestDescent, optimization.NewtonMethod]:

                #DEFAULT
                min_val = -2
                max_val = 2
                grid_size = 0.2
                ranges = arange(min_val, max_val, grid_size)

                starting_positions = linspace(min_val + stepsize, max_val - stepsize, 10)


                fitness_function = fitness_function_name(ranges = ranges)
                optimizer = optimization.HillClimber(
                                        fitness_function = fitness_function,
                                        precision = precision,
                                        max_iterations = max_iterations,
                                        path = optimizer_output_path,
                                        individuals_data = individuals_data,
                                        fitness_statistics = fitness_statistics,
                                        starting_position = starting_positions[0])

                if fitness_function_name.__name__ == "SquaredError2D":
                    min_val = -2
                    max_val = 2
                    grid_size = 0.2
                    ranges =  arange(min_val, max_val, grid_size)
                    starting_positions = linspace(min_val + stepsize, max_val - stepsize, 10)

                if fitness_function_name.__name__ == "Trimodal2D":
                    min_val = -3.4
                    max_val = -0.6
                    grid_size = 0.2
                    ranges =  arange(min_val, max_val, grid_size)
                    starting_positions = linspace(min_val + stepsize, max_val - stepsize, 10)

                if fitness_function_name.__name__ == "Plateau3D":
                    min_val = -2
                    max_val = 2
                    grid_size = 0.5
                    ranges =  arange(min_val, max_val, grid_size)
                    starting_positions = linspace(min_val + stepsize, max_val - stepsize, 10)
                    starting_positions =  product(starting_positions, repeat=2)
                    starting_positions = [np.asarray(val) for val in starting_positions]
                    n = 10

                    #not a circle
                    # starting_positions = [(math.cos(2*pi/n*x),math.sin(2*pi/n*x)) for x in xrange(0,n)]
                    # starting_positions = [np.asarray(x) for x in starting_positions]

                fitness_function = fitness_function_name(ranges = ranges)

                if optimizer_name.__name__ == "HillClimber":
                    for sp in starting_positions:
                        optimizer = optimizer_name(
                                                    fitness_function = fitness_function,
                                                    precision = precision,
                                                    max_iterations = max_iterations,
                                                    path = optimizer_output_path,
                                                    individuals_data = individuals_data,
                                                    fitness_statistics = fitness_statistics,
                                                    starting_position = sp)

                        _max, _min, _mean =  optimizer.run()

                        # aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                        #                           fitstat_data=optimizer_output_path + fitness_statistics,
                        #                           fitscape_data=fitness_function.get_fitness_filename(),
                        #                           ranges=ranges,
                        #                           max_iterations=optimizer.get_current_iteration(),
                        #                           width=width, height=height, dpi=dpi,
                        #                           dim=fitness_function.get_dimensionality())
                        # aw.export_jpg()
                        #
                        # fname = optimizer_name.__name__ + "/" + fitness_function_name.__name__
                        # fname = fname +  "_" + array_str(around(sp, 2))
                        # print fname
                        #
                        # rename("export.jpg", "output/images/" + fname)

                        ds.append({
                                    'min':_min.astype(float),
                                    'mean':_mean.astype(float),
                                    'max':_max.astype(float),
                                    'ff':fitness_function_name.__name__,
                                    'opt':optimizer_name.__name__,
                                    'sp':sp.astype(float)
                                    })


                if optimizer_name.__name__ == "SteepestDescent":
                    for sp in starting_positions:
                        for learning_rate in test_vals:
                            for inertia in test_vals:
                                optimizer = optimization.SteepestDescent(
                                                        fitness_function = fitness_function,
                                                        precision = precision,
                                                        max_iterations = max_iterations,
                                                        path = optimizer_output_path,
                                                        individuals_data = individuals_data,
                                                        fitness_statistics = fitness_statistics,
                                                        starting_position = sp,
                                                        learning_rate = learning_rate,
                                                        inertia = inertia)

                                _max, _min, _mean =  optimizer.run()


                                # aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                                #                           fitstat_data=optimizer_output_path + fitness_statistics,
                                #                           fitscape_data=fitness_function.get_fitness_filename(),
                                #                           ranges=ranges,
                                #                           max_iterations=optimizer.get_current_iteration(),
                                #                           width=width, height=height, dpi=dpi,
                                #                           dim=fitness_function.get_dimensionality())
                                # aw.export_jpg()
                                #
                                # fname = optimizer_name.__name__ + "/" + fitness_function_name.__name__
                                # fname = fname +  "_" + array_str(around(sp, 2))
                                # fname = fname +  "_" + array_str(around(learning_rate, 2))
                                # fname = fname +  "_" + array_str(around(inertia, 2))
                                # print fname
                                #
                                # rename("export.jpg", "output/images/" + fname)

                                ds.append({
                                            'min':_min.astype(float),
                                            'mean':_mean.astype(float),
                                            'max':_max.astype(float),
                                            'learning_rate':learning_rate.astype(float),
                                            'inertia':inertia.astype(float),
                                            'ff':fitness_function_name.__name__,
                                            'opt':optimizer_name.__name__,
                                            'sp':sp.astype(float)
                                            })

                if optimizer_name.__name__ == "NewtonMethod" and fitness_function_name.__name__ != "Plateau3D" :
                    for sp in starting_positions:
                        optimizer = optimization.NewtonMethod(
                                                fitness_function = fitness_function,
                                                precision = precision,
                                                max_iterations = max_iterations,
                                                path = optimizer_output_path,
                                                individuals_data = individuals_data,
                                                fitness_statistics = fitness_statistics,
                                                starting_position = sp)
                        _max, _min, _mean =  optimizer.run()


                        # aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                        #                           fitstat_data=optimizer_output_path + fitness_statistics,
                        #                           fitscape_data=fitness_function.get_fitness_filename(),
                        #                           ranges=ranges,
                        #                           max_iterations=optimizer.get_current_iteration(),
                        #                           width=width, height=height, dpi=dpi,
                        #                           dim=fitness_function.get_dimensionality())
                        # aw.export_jpg()
                        #
                        # fname = optimizer_name.__name__ + "/" + fitness_function_name.__name__
                        # fname = fname +  "_" + array_str(around(sp, 2))
                        # print fname
                        #
                        # rename("export.jpg", "output/images/" + fname)

                        ds.append({
                                    'min':_min.astype(float),
                                    'mean':_mean.astype(float),
                                    'max':_max.astype(float),
                                    'ff':fitness_function_name.__name__,
                                    'opt':optimizer_name.__name__,
                                    'sp':sp.astype(float)
                                    })

    #ANALYSIS RUN
    latest_analysis()

    #DATA EXPORT
    keys = [
        'min',
        'mean',
        'max',
        'learning_rate',
        'inertia',
        'ff',
        'opt',
        'sp']

    #Export Data into data.csv
    with open('data.csv', 'wb') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(ds)

    min_val = -2
    max_val = 2
    grid_size = 0.5
    ranges =  arange(min_val, max_val, grid_size)
    # starting_positions =  product(linspace(-1.0, 1.0, 3), repeat=2)
    n = 10

    # starting_positions = [(math.cos(2*pi/n*x),math.sin(2*pi/n*x)) for x in xrange(0,n)]
    starting_positions = [(0.81, -0.59)]
    starting_positions = [np.asarray(x) for x in starting_positions]

   # SETUP FITNESS FUNCTIONs
    # fitness_function = fitness_factory.SquaredError2D(ranges = ranges)
    # fitness_function = fitness_factory.Trimodal2D(ranges = ranges)

    fitness_function = fitness_factory.Plateau3D(ranges = ranges)

    # SETUP OPTIMIZATIONS
    optimizer = optimization.HillClimber(
                                                fitness_function = fitness_function,
                                                precision = precision,
                                                stepsize = stepsize,
                                                max_iterations = max_iterations,
                                                path = optimizer_output_path,
                                                individuals_data = individuals_data,
                                                fitness_statistics = fitness_statistics,
                                                # starting_position = np.array([-1.0])
                                                starting_position = starting_positions[0]
                                                )

    # optimizer = optimization.SteepestDescent(
    #                                             fitness_function = fitness_function,
    #                                             precision = precision,
    #                                             max_iterations = max_iterations,
    #                                             path = optimizer_output_path,
    #                                             individuals_data = individuals_data,
    #                                             fitness_statistics = fitness_statistics,
    #                                             starting_position = np.array([0.0, -1.6]),
    #                                             # starting_position = np.array([1.0]),
    #
    #                                             learning_rate = 0.01,
    #                                             inertia = 1.0# no inertia means no momentum
    #                                             )

    # optimizer = optimization.NewtonMethod(
    #                                             fitness_function = fitness_function,
    #                                             precision = precision,
    #                                             max_iterations = max_iterations,
    #                                             path = optimizer_output_path,
    #                                             individuals_data = individuals_data,
    #                                             fitness_statistics = fitness_statistics,
    #                                             starting_position = -0.6,
    #                                             )
    # # RUN OPTIMIZATION
    maxfit, minfit, meanfit = optimizer.run()

    # VISUALIZATION
    aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                              fitstat_data=optimizer_output_path + fitness_statistics,
                              fitscape_data=fitness_function.get_fitness_filename(),
                              ranges=ranges,
                              max_iterations=optimizer.get_current_iteration(),
                              width=width, height=height, dpi=dpi,
                              dim=fitness_function.get_dimensionality())
    aw.export_jpg()
    aw.show()

    sys.exit(qApp.exec_())

if __name__ == "__main__":
    main()
