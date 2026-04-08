#----------------------------------------------------------------------------------
# Experiment 1: Comparison of Sequential and Threaded PSO
# @author: David Ortega Lozano
# @date: 2026-03-11
# @version: 1.0
# @description: This code runs an experiment to compare the performance of the 
# sequential and threaded versions of the Particle Swarm Optimization (PSO) 
# algorithm. It uses the sphere function as the objective function to minimize and 
# measures the best position, best value, and execution time for both methods. The 
# results are displayed in a table format for easy comparison.
#----------------------------------------------------------------------------------

import time
from core.swarm import Swarm
from prettytable import PrettyTable
import numpy as np
import json

with open('config.json') as config_file:
    config = json.load(config_file)

# The compare function runs the PSO algorithm using both sequential and threaded
# methods for a given number of dimensions and objective function choice. It
# initializes a table to store the results, creates a swarm of particles, optimizes
# the objective function, and registers the results in the table. Finally, it 
# prints the table with the results for both methods, showing the best position, 
# best value,
def compare(dimensions, 
            function_choice, 
            methods = [1, 2]
            ) -> None:

    # Initialize the table of results
    table = init_table(dimensions)

    for method in methods:
        # Create the swarm of particles
        swarm = Swarm(config["PARTICLES"],
                      function_choice, 
                      dimensions)

        # Optimize the objective function
        start = time.time()
        swarm.optimize(method)
        end = time.time()
        execution_time = end - start

        # Include results in the table
        table = register_results(table, 
                                 method, 
                                 swarm.global_best_position, 
                                 swarm.global_best_value, 
                                 execution_time, 
                                 dimensions)
    
    # Print final results
    print(table)

# The init_table function initializes a PrettyTable object with appropriate column
# names based on the number of dimensions. If the dimensions are greater than 3,
# it only includes the method, best value, and execution time. Otherwise, it also
# includes the best position.
def init_table(dimensions) -> PrettyTable:
    table = PrettyTable()
    if dimensions > 3:
        table.field_names = ["Method",
                             "Best Value", 
                             "Execution Time (s)"]
    else:
        table.field_names = ["Method", 
                             "Best Position", 
                             "Best Value", 
                             "Execution Time (s)"]
    return table

# The register_results function takes the results of the optimization process and 
# adds them to the table. It formats the best position, best value, and execution 
# time for better readability and determines the method name for display based on 
# the method used (sequential or threading). Depending on the number of dimensions, 
# it either includes the best position in the table or omits it for higher 
# dimensions.
def register_results(table, 
                     method, 
                     best_position, 
                     best_value, 
                     execution_time, 
                     dimensions
                     ) -> PrettyTable:
    
    # Determine the method name for display between Sequential, Threading and Multiprocessing
    method_text = "Sequential" if method == 1 else "Threading" if method == 2 else "Multiprocessing"
    
    # Round the best value, best position, and execution time for better readability
    value = round(best_value, config["NUM_DECIMALS"])
    position = np.round(best_position, config["NUM_DECIMALS"])
    ex_time = round(execution_time, config["NUM_DECIMALS"])

    # Add the results to the table, showing only the best position if dimensions are equal or lower than 3
    if dimensions > 3:
        table.add_row([method_text, value, ex_time])
    else:
        table.add_row([method_text, position.tolist(), value, ex_time]) 
    
    # Return the updated table
    return table