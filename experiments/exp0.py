#----------------------------------------------------------------------------------
# Experiment 0: Manual Run of PSO
# @author: David Ortega Lozano
# @date: 2026-03-18
# @version: 1.0
# @description: This code allows for a manual run of the Particle Swarm 
# Optimization (PSO) algorithm with user-defined parameters. The user can select 
# the objective function to minimize, the number of particles, dimensions, 
# iterations, and the optimization method (sequential or threading).The results are
# displayed in a table format for easy comparison.
#----------------------------------------------------------------------------------

import time
import json
from parallel.v0_pso_sequential import Secuential
from parallel.v1_pso_threading import Threading
from experiments.exp1 import init_table, register_results

with open('config.json') as config_file:
    config = json.load(config_file)

# The ask_value function prompts the user to select an value of a specific label 
# (e.g., particles, dimensions, iterations) and returns the choice.
def ask_value(label) -> int:
    choice = input(f"Enter the number of {label}: ")
    return int(choice)

# The ask_method function prompts the user to select the optimization method 
# (sequential or threading) and returns the choice.
def ask_method() -> int:
    while True:
        print("Select the optimization method:")
        print("1. Sequential")
        print("2. Threading")
        choice = input("Enter the number of the method: ")
        if choice in ['1', '2']:
            return int(choice)
        else:
            print("Invalid choice. Please enter 1 or 2.")

# The manual_run function runs the PSO algorithm with user-defined parameters and 
# displays the results in a table format. It creates a swarm of particles, 
# optimizes the objective function, and prints the best solution found along with 
# the execution time.
def manual_run(function_choice) -> None:
    
    # Ask the user for the number of particles and dimensions, and the maximum 
    # number of iterations.
    num_particles = ask_value("particles")
    num_dimensions = ask_value("dimensions")
    max_iterations = ask_value("iterations")

    # Determine the optimization method based on the user's choice and create the
    # swarm accordingly.
    method = ask_method()
    match method:
        case 1:
            swarm = Secuential(num_particles, 
                               function_choice, 
                               num_dimensions)
        case 2:
            swarm = Threading(num_particles, 
                              function_choice, 
                              num_dimensions)
            
    # Measure the execution time of the optimization process.
    start = time.time()
    swarm.optimize()
    end = time.time()
    execution_time = end - start

    # Print the best solution found
    table = init_table(num_dimensions)
    table = register_results(table, 
                             method, 
                             swarm.global_best_position, 
                             swarm.global_best_value, 
                             execution_time, 
                             num_dimensions)
    print(table)