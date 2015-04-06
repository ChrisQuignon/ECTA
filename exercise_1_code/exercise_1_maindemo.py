#!/usr/bin/env python
# Example main

from visualize_results  import *
import fitness_factory
import exercise_1_optimization as optimization
from numpy import arange, linspace, logspace
from os import rename
import numpy as np
from itertools import product

def main():
    qApp = QtGui.QApplication(sys.argv)
    logging.basicConfig(level=logging.INFO)

    # Window configuration
    width = 3
    height = 3
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

    def new_analysis():

        # test_vals = [0.0, 0.001, 0.01, 0.01, 0.1, 0.2, 0.4, 0.8, 1.0, 1.2]
        test_vals = np.hstack(([0], logspace(0, 2, 9)/250))

        starting_positions_2D = linspace(min_val + stepsize, max_val - stepsize, 10)
        starting_positions_3D = product(linspace(-1.0, 1.0, 3), repeat=2)

        print test_vals

        #HillClimber2D
        for starting_position in starting_positions_2D:
            for fitness_function_name in [fitness_factory.SquaredError2D, fitness_factory.Trimodal2D]:
                fitness_function = fitness_function_name(ranges = ranges)

                optimizer = optimization.HillClimber2DLAB(
                                        fitness_function = fitness_function,
                                        precision = precision,
                                        max_iterations = max_iterations,
                                        path = optimizer_output_path,
                                        individuals_data = individuals_data,
                                        fitness_statistics = fitness_statistics,
                                        starting_position = np.array([starting_position]))
                optimizer.run()
                aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                                          fitstat_data=optimizer_output_path + fitness_statistics,
                                          fitscape_data=fitness_function.get_fitness_filename(),
                                          ranges=ranges,
                                          max_iterations=optimizer.get_current_iteration(),
                                          width=width, height=height, dpi=dpi,
                                          dim=fitness_function.get_dimensionality())
                aw.export_jpg()

                fname = "HillClimber/"
                fname = fname + fitness_function_name.__name__
                fname = fname +  "_" + str("%.2f" % round(starting_position,2))

                print fname

                rename("export.jpg", "output/images/" + fname + "_steps")
                rename("export_fitness.jpg", "output/images/" + fname + "_fitness")

        # HillClimber3D
        fitness_function = fitness_factory.Plateau3D(ranges = ranges)
        for x, y in starting_positions_3D:
            optimizer = optimization.HillClimber2DLAB(
                                    fitness_function = fitness_function,
                                    precision = precision,
                                    max_iterations = max_iterations,
                                    path = optimizer_output_path,
                                    individuals_data = individuals_data,
                                    fitness_statistics = fitness_statistics,
                                    starting_position = np.array([x, y]))
            optimizer.run()
            aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                                      fitstat_data=optimizer_output_path + fitness_statistics,
                                      fitscape_data=fitness_function.get_fitness_filename(),
                                      ranges=ranges,
                                      max_iterations=optimizer.get_current_iteration(),
                                      width=width, height=height, dpi=dpi,
                                      dim=fitness_function.get_dimensionality())
            aw.export_jpg()

            fname = "HillClimber/"
            fname = fname + "Plateau3D"
            fname = fname +  "_" + str("%.2f" % round(x,2)) + "," + str("%.2f" % round(y,2))

            print fname

            rename("export.jpg", "output/images/" + fname + "_steps")
            rename("export_fitness.jpg", "output/images/" + fname + "_fitness")

        # SteepestDescent
        for starting_position in starting_positions_2D:
            for fitness_function_name in [fitness_factory.SquaredError2D, fitness_factory.Trimodal2D]:
                fitness_function = fitness_function_name(ranges = ranges)

                for learning_rate in test_vals:
                    for inertia in test_vals:

                        optimizer = optimization.SteepestDescent(
                                                fitness_function = fitness_function,
                                                precision = precision,
                                                max_iterations = max_iterations,
                                                path = optimizer_output_path,
                                                individuals_data = individuals_data,
                                                fitness_statistics = fitness_statistics,
                                                starting_position = np.array([starting_position]),
                                                learning_rate = learning_rate,
                                                inertia = inertia)
                        optimizer.run()
                        aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                                                  fitstat_data=optimizer_output_path + fitness_statistics,
                                                  fitscape_data=fitness_function.get_fitness_filename(),
                                                  ranges=ranges,
                                                  max_iterations=optimizer.get_current_iteration(),
                                                  width=width, height=height, dpi=dpi,
                                                  dim=fitness_function.get_dimensionality())
                        aw.export_jpg()

                        fname = "SteepestDescent/"
                        fname = fname + fitness_function_name.__name__
                        fname = fname +  "_" + str("%.2f" % round(starting_position,2))
                        fname = fname +  "_" + str("%.2f" % round(learning_rate,2))
                        fname = fname +  "_" + str("%.2f" % round(inertia,2))

                        print fname

                        rename("export.jpg", "output/images/" + fname + "_steps")
                        rename("export_fitness.jpg", "output/images/" + fname + "_fitness")

        # SteepestDescent3D
        fitness_function = fitness_factory.Plateau3D(ranges = ranges)
        for x, y in starting_positions_3D:
            for learning_rate in test_vals:
                for inertia in test_vals:
                    optimizer = optimization.SteepestDescent(
                                            fitness_function = fitness_function,
                                            precision = precision,
                                            max_iterations = max_iterations,
                                            path = optimizer_output_path,
                                            individuals_data = individuals_data,
                                            fitness_statistics = fitness_statistics,
                                            starting_position = np.array([x, y]),
                                            learning_rate = learning_rate,
                                            inertia = inertia)
                    optimizer.run()
                    aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                                              fitstat_data=optimizer_output_path + fitness_statistics,
                                              fitscape_data=fitness_function.get_fitness_filename(),
                                              ranges=ranges,
                                              max_iterations=optimizer.get_current_iteration(),
                                              width=width, height=height, dpi=dpi,
                                              dim=fitness_function.get_dimensionality())
                    aw.export_jpg()

                    fname = "SteepestDescent/"
                    fname = fname + "Plateau3D"
                    fname = fname +  "_" + str("%.2f" % round(x,2)) + "," + str("%.2f" % round(y,2))
                    fname = fname +  "_" + str("%.2f" % round(learning_rate,2))
                    fname = fname +  "_" + str("%.2f" % round(inertia,2))

                    print fname

                    rename("export.jpg", "output/images/" + fname + "_steps")
                    rename("export_fitness.jpg", "output/images/" + fname + "_fitness")


        # NewtonMethod
        for starting_position in starting_positions_2D:
            for fitness_function_name in [fitness_factory.SquaredError2D, fitness_factory.Trimodal2D]:
                fitness_function = fitness_function_name(ranges = ranges)

                for learning_rate in test_vals:
                    for inertia in test_vals:

                        optimizer = optimization.NewtonMethod(
                                                fitness_function = fitness_function,
                                                precision = precision,
                                                max_iterations = max_iterations,
                                                path = optimizer_output_path,
                                                individuals_data = individuals_data,
                                                fitness_statistics = fitness_statistics,
                                                starting_position = np.array([starting_position]))
                        optimizer.run()
                        aw = EAVApplicationWindow(individuals_data=optimizer_output_path + individuals_data,
                                                  fitstat_data=optimizer_output_path + fitness_statistics,
                                                  fitscape_data=fitness_function.get_fitness_filename(),
                                                  ranges=ranges,
                                                  max_iterations=optimizer.get_current_iteration(),
                                                  width=width, height=height, dpi=dpi,
                                                  dim=fitness_function.get_dimensionality())
                        aw.export_jpg()

                        fname = "NewtonMethod/"
                        fname = fname + fitness_function_name.__name__
                        fname = fname +  "_" + str("%.2f" % round(starting_position,2))

                        print fname

                        rename("export.jpg", "output/images/" + fname + "_steps")
                        rename("export_fitness.jpg", "output/images/" + fname + "_fitness")

    #ANALYSIS RUN
    new_analysis()

    # TEST
    # SETUP FITNESS FUNCTIONs
    # fitness_function = fitness_factory.SquaredError2D(ranges = ranges)
    # fitness_function = fitness_factory.Trimodal2D(ranges = ranges)


    fitness_function = fitness_factory.Plateau3D(ranges = ranges)

    # SETUP OPTIMIZATIONS
    # optimizer = optimization.HillClimber2DLAB(
    #                                             fitness_function = fitness_function,
    #                                             precision = precision,
    #                                             stepsize = stepsize,
    #                                             max_iterations = max_iterations,
    #                                             path = optimizer_output_path,
    #                                             individuals_data = individuals_data,
    #                                             fitness_statistics = fitness_statistics,
    #                                             starting_position = np.array([0.0, 0.0])
    #                                             # starting_position = np.array([1.0])
    #                                             )

    optimizer = optimization.SteepestDescent(
                                                fitness_function = fitness_function,
                                                precision = precision,
                                                max_iterations = max_iterations,
                                                path = optimizer_output_path,
                                                individuals_data = individuals_data,
                                                fitness_statistics = fitness_statistics,
                                                starting_position = np.array([0.0, -1.6]),
                                                # starting_position = np.array([1.0]),

                                                learning_rate = 0.01,
                                                inertia = 1.0# no inertia means no momentum
                                                )

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
