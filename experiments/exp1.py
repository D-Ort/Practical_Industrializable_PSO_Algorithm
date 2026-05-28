#----------------------------------------------------------------------------------
# Experiment 1: Comparison of Sequential, Threaded and Multiprocess PSO
# @author: David Ortega Lozano
# @date: 2026-03-11
# @version: 2.0
# @description: This code runs an experiment to compare the performance of the 
# sequential, threaded and multiprocess versions of the Particle Swarm Optimization (PSO) 
# algorithm. It uses a objective function as the objective function to minimize and 
# measures the best position, best value, and execution time for both methods. The 
# results are displayed in a table format for easy comparison.
#----------------------------------------------------------------------------------

import time
from parallel.v0_pso_sequential import Secuential
from parallel.v1_pso_threading import Threading
from parallel.v2_pso_multiprocessing import Multiprocess
from parallel.v3_pso_asyncIO import AsyncIO
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
            methods = list(range(1, len(config["METHODS"]) + 1))
            ) -> None:

    # Initialize the table of results and the seeds for reproducibility
    table = init_table(dimensions)
    p_seeds = [config["RANDOM_SEED"] + i for i in range(config["PARTICLES"])]

    # The number of dimensions, particles and iterations are ensured to be 3, 30 
    # and 10 for the Industrial case function, as it is a 3-dimensional function
    # with higher complexity. For other functions, the dimensions are determined 
    # by the input parameter and the particles and iterations are determined by
    # the config file. The count variable that is used to assign a unique 
    # experiment ID, is initialized to 0.
    count = 0
    if function_choice == 5:
        dimensions = 3
        particles = 30
        iterations = 10
    else:
        particles = config["PARTICLES"]
        iterations = config["ITERATIONS"]
    

    for method in methods:
        # Create the swarm of particles
        match method:
            case 1:
                swarm = Secuential(particles,
                                   function_choice,
                                   p_seeds,
                                   dimensions,
                                   exp_id=count,
                                   num_iterations=iterations)
            case 2:
                swarm = Threading(particles,
                                  function_choice,
                                  p_seeds,
                                  dimensions,
                                  exp_id=count,
                                  num_iterations=iterations)

            case 3:
                swarm = Multiprocess(particles,
                                     function_choice,
                                     p_seeds,
                                     dimensions,
                                     exp_id=count,
                                     num_iterations=iterations)
            case 4:
                swarm = AsyncIO(particles,
                                function_choice,
                                p_seeds,
                                dimensions,
                                exp_id=count,
                                num_iterations=iterations)
            case _:
                swarm = Secuential(particles,
                                   function_choice,
                                   p_seeds,
                                   dimensions,
                                   exp_id=count,
                                   num_iterations=iterations)
                print("Error: Invalid method:", method)

        # Optimize the objective function
        start = time.time()
        swarm.optimize()
        end = time.time()
        execution_time = end - start

        # Include results in the table
        if(function_choice == 5):

            hiperparameters = []
            hiperparameters.append(np.clip(abs(swarm.global_best_position[0]), 0.0001, 100))
            hiperparameters.append(np.clip(abs(swarm.global_best_position[1]), 50, 1000))
            hiperparameters.append(np.clip(abs(swarm.global_best_position[2]), 1e-6, 1e-1))
            
            table = register_results(table, 
                                     method, 
                                     hiperparameters, 
                                     swarm.global_best_value, 
                                     execution_time, 
                                     dimensions)
        else:
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
    method_text = config["METHODS"][method - 1] if method in range(1, len(config["METHODS"]) + 1) else method
    
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