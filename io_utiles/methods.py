#----------------------------------------------------------------------------------
# Methods: tools to save the logs of the particles in a file and to clean the 
# environment by removing the logs file if it exists.
# @author: David Ortega Lozano
# @date: 2026-03-19
# @version: 1.0
# @description: This module contains methods to save the logs of the particles in a
# file and to clean the environment by removing the logs file if it exists.
#----------------------------------------------------------------------------------
import csv
import os
import numpy as np
import json

with open('config.json') as config_file:
    config = json.load(config_file)

# The save_logs method takes the swarm and method as parameters and 
# saves the logs of each particle in the specified file.
def save_logs(swarm, 
              method
              ) -> None:

    iterations = len(swarm.particles[0].logs)

    # The number of dimensions is determined by the length of the position vector 
    # of the first particle
    dim = len(np.array(swarm.particles[0].position).flatten())
    dimensions = [f'pos_{i}' for i in range(dim)]
    best_dimensions = [f'best_pos_{i}' for i in range(dim)]
    header = (['iteration'] + 
              dimensions + 
              ['value'] + 
              best_dimensions + 
              ['best_value'] + 
              ['method'])
    method_label = "Sequential" if method == 1 else "Threading"

    # The average position in each dimension of all particles for each iteration is
    # calculated and stored in the average list, which is then saved in the file.
    average_pos, average_value, list_global_pos, list_global_value = calc_iteration_average(swarm, iterations)

    # The header is written only if the file does not exist
    try:
        with open(config["LOGS_FILE_NAME"], 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    except FileExistsError:
        pass

    # Now write the data
    with open(config["LOGS_FILE_NAME"], 'a', newline='') as f:
        writer = csv.writer(f)
        for i in range(iterations):
            row = [i+1]
            row += list(np.array(average_pos[i]).flatten())
            row.append(average_value[i])

            row += list(np.array(list_global_pos[i]).flatten())
            row.append(list_global_value[i])

            row.append(method_label)
            writer.writerow(row)

# The clean_logs method checks if the logs file exists and removes it to ensure a 
# clean environment for new experiments.
def clean_logs() -> None:
    if os.path.exists(config["LOGS_FILE_NAME"]):
        os.remove(config["LOGS_FILE_NAME"])

# The calc_iteration_average method calculates the average position in each 
# dimension, the average value, the list of global best positions, and the list of 
# global best values for each iteration and returns thems. It takes the swarm, 
# number of decimals, and maximum iterations as parameters.
def calc_iteration_average(swarm, 
                           iterations
                           ) -> tuple:
    
    # Initialize lists to store the average position, average value, global best
    # positions, and global best values for each iteration to be calculated and 
    # returned.
    average_pos = []
    average_value = []
    list_global_pos = []
    list_global_value = []

    for i in range(iterations):
        # The first flag is used to initialize the maximum position and value with 
        # the values of the first particle in the first iteration, and then it is 
        # set to False to avoid reinitializing them in the next iterations. The 
        # average position and average value are calculated by summing the 
        # positions and values of all particles and then dividing by the number of 
        # particles. The global best position and value are updated if a better 
        # value is found among the particles in the current iteration.
        first = True
        position_avg = []
        value_avg = 0
        max_position = []
        max_value = 0

        for particle in swarm.particles:

            position, value, best_position, best_value = particle.logs[i]
            position_avg.append(position)
            value_avg += value

            if first or value < max_value:
                max_position = best_position
                max_value = best_value
                first = False

        average_pos.append(np.round(np.mean(position_avg, axis=0), config['NUM_DECIMALS']))
        average_value.append(round(value_avg / swarm.num_particles, config['NUM_DECIMALS']))
        list_global_pos.append(np.round(max_position, config['NUM_DECIMALS']))
        list_global_value.append(round(max_value, config['NUM_DECIMALS']))

    return average_pos, average_value, list_global_pos, list_global_value