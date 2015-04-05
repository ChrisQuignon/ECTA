#!/usr/bin/env python
# Example main

from visualize_results  import *
import fitness_factory
import exercise_1_optimization as optimization
from numpy import arange, linspace
from os import rename
import numpy as np

def main():
    qApp = QtGui.QApplication(sys.argv)
    logging.basicConfig(level=logging.INFO)

    # Window configuration
    width = 16
    height = 20
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

    def analysis():
        optimizers = [optimization.HillClimber2DLAB, optimization.SteepestDescent, optimization.NewtonMethod]
        fitness_functions = [fitness_factory.SquaredError2D, fitness_factory.Trimodal2D]
        #fitness_functions.append(fitness_factory.Plateau3D)

        for optimizer_name in optimizers:
            for fitness_function_name in fitness_functions:
                fitness_function = fitness_function_name(ranges = ranges)

                for starting_position in linspace(min_val + stepsize, max_val - stepsize, 10):

                    if optimizer_name.__name__ == "SteepestDescent":
                        for learning_rate in linspace(0, 1, 10):
                            for inertia in linspace(0, 1, 10):
                                optimizer = optimizer_name(
                                                        fitness_function = fitness_function,
                                                        precision = precision,
                                                        max_iterations = max_iterations,
                                                        path = optimizer_output_path,
                                                        individuals_data = individuals_data,
                                                        fitness_statistics = fitness_statistics,
                                                        starting_position = starting_position,
                                                        learning_rate = learning_rate,
                                                        inertia = inertia)
                                optimizer.run()

                                aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                                                          fitstat_data=optimizer_output_path + fitness_statistics,
                                                          fitscape_data=fitness_function.get_fitness_filename(),
                                                          ranges=ranges, max_iterations=optimizer.get_current_iteration(),
                                                          width=width, height=height, dpi=dpi,
                                                          dim=fitness_function.get_dimensionality())
                                aw.export_jpg()

                                # fname = "/images/"
                                fname = optimizer_name.__name__
                                fname = fname + "_" + fitness_function_name.__name__
                                fname = fname +  "_" + str("%.2f" % round(starting_position,2))
                                fname = fname +  "_" + str("%.2f" % round(learning_rate,2))
                                fname = fname +  "_" + str("%.2f" % round(inertia,2))
                                # fname = fname + ".jpg"

                                print fname

                                rename("export.jpg", "output/images/" + fname + "_steps")
                                rename("export_fitness.jpg", "output/images/" + fname + "_fitness")
                    else:

                        optimizer = optimizer_name(
                                                fitness_function = fitness_function,
                                                precision = precision,
                                                max_iterations = max_iterations,
                                                path = optimizer_output_path,
                                                individuals_data = individuals_data,
                                                fitness_statistics = fitness_statistics,
                                                starting_position = starting_position)
                        optimizer.run()

                        aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                                                  fitstat_data=optimizer_output_path + fitness_statistics,
                                                  fitscape_data=fitness_function.get_fitness_filename(),
                                                  ranges=ranges, max_iterations=optimizer.get_current_iteration(),
                                                  width=width, height=height, dpi=dpi,
                                                  dim=fitness_function.get_dimensionality())
                        aw.export_jpg()

                        #fname = "/images/"
                        fname = optimizer_name.__name__
                        fname = fname + "_" + fitness_function_name.__name__
                        fname = fname +  "_" + str("%.2f" % round(starting_position,2))
                        # fname = fname + ".jpg"

                        print fname

                        rename("export.jpg", "output/images/" + fname + "_steps")
                        rename("export_fitness.jpg", "output/images/" + fname + "_fitness")

    #ANALYSIS RUN
    #analysis()

    #TEST
    # SETUP FITNESS FUNCTIONs
    # fitness_function = fitness_factory.SquaredError2D(ranges = ranges)
    # fitness_function = fitness_factory.Trimodal2D(ranges = ranges)

    # TODO: Fix error
    fitness_function = fitness_factory.Plateau3D(ranges = ranges)

    # SETUP OPTIMIZATIONS
    optimizer = optimization.HillClimber2DLAB(
                                                fitness_function = fitness_function,
                                                precision = precision,
                                                stepsize = stepsize,
                                                max_iterations = max_iterations,
                                                path = optimizer_output_path,
                                                individuals_data = individuals_data,
                                                fitness_statistics = fitness_statistics,
                                                starting_position = np.array([0.0, 0.0])
                                                # starting_position = np.array([1.0])
                                                )

    # optimizer = optimization.SteepestDescent(
    #                                             fitness_function = fitness_function,
    #                                             precision = precision,
    #                                             max_iterations = max_iterations,
    #                                             path = optimizer_output_path,
    #                                             individuals_data = individuals_data,
    #                                             fitness_statistics = fitness_statistics,
    #                                             starting_position = np.array([1.0, 0.0]),
    #                                             # starting_position = np.array([-1.5]),
    #
    #                                             learning_rate = 0.1,
    #                                             inertia = 0.1# no inertia means no momentum
    #                                             )

    # optimizer = optimization.newtonMethod(
    #                                             fitness_function = fitness_function,
    #                                             precision = precision,
    #                                             max_iterations = max_iterations,
    #                                             path = optimizer_output_path,
    #                                             individuals_data = individuals_data,
    #                                             fitness_statistics = fitness_statistics,
    #                                             starting_position = -0.6,
    #                                             )

    # RUN OPTIMIZATION
    optimizer.run()



    # VISUALIZATION
    aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                              fitstat_data=optimizer_output_path + fitness_statistics,
                              fitscape_data=fitness_function.get_fitness_filename(),
                              ranges=ranges, max_iterations=optimizer.get_current_iteration(),
                              width=width, height=height, dpi=dpi,
                              dim=fitness_function.get_dimensionality())

    aw.show()

    aw.export_jpg()

    sys.exit(qApp.exec_())

if __name__ == "__main__":
    main()
