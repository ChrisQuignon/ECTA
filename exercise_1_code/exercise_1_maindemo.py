#!/usr/bin/env python
# Example main

from visualize_results  import *
import fitness_factory
import exercise_1_optimization as optimization
from numpy import arange

def main():
    qApp = QtGui.QApplication(sys.argv)
    logging.basicConfig(level=logging.INFO)

    # Window configuration
    width = 8
    height = 10
    dpi = 100

    # Data file and path definitions
    optimizer_output_path= 'output/optimizer/'
    fitness_statistics = 'fitness_statistics'
    individuals_data = 'individuals'

    # Ranges (both for fitness landscape visualization as well as optimization boundaries
    min_val = -2
    max_val = 2
    grid_size = 0.05
    ranges = arange(min_val, max_val, grid_size)

    # Optimization parameters
    max_iterations = 50
    stepsize = 0.1
    learning_rate = 0.1
    precision = 0.0001


    # SETUP FITNESS FUNCTION
    fitness_function = fitness_factory.SquaredError2D(ranges = ranges)
    #fitness_function = fitness_factory.Trimodal2D(ranges = ranges)

    #TODO: Fix error
    #fitness_function = fitness_factory.Plateau3D(ranges = ranges)

    # SETUP OPTIMIZATION
    optimizer = optimization.HillClimber2DLAB(
                                                fitness_function = fitness_function,
                                                precision = precision,
                                                stepsize = stepsize,
                                                max_iterations = max_iterations,
                                                path = optimizer_output_path,
                                                individuals_data = individuals_data,
                                                fitness_statistics = fitness_statistics,
                                                starting_position = -1.6
                                                )

    # optimizer = optimization.steepestDescent(
    #                                             fitness_function = fitness_function,
    #                                             precision = precision,
    #                                             max_iterations = max_iterations,
    #                                             path = optimizer_output_path,
    #                                             individuals_data = individuals_data,
    #                                             fitness_statistics = fitness_statistics,
    #
    #                                             starting_position = -0.6,
    #                                             learning_rate = 0.1,
    #                                             inertia = 0.8# no inertia means no momentum
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

    sys.exit(qApp.exec_())

if __name__ == "__main__":
    main()
