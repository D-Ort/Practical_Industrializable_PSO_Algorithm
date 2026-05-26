#----------------------------------------------------------------------------------
# Methods: tools to save the logs of the particles in a file and to clean the 
# environment by removing the logs file if it exists.
# @author: David Ortega Lozano
# @date: 2026-03-19
# @version: 1.2
# @description: This module contains methods to save the logs of the particles in a
# file and to clean the environment by removing the logs file if it exists.
#----------------------------------------------------------------------------------
import csv
import os
import numpy as np
import json

with open('config.json') as config_file:
    config = json.load(config_file)

# The save_logs method first determines the number of dimensions from the swarm, 
# and then it constructs the header for the CSV file, which includes the experiment
# ID, method, function name, dimensions, iteration, particle ID, positions, values,
# best positions, and best values. It checks if the logs file already exists, and 
# if not, it creates the file and writes the header. Finally, it appends the logs 
# of each particle to the file, including the position, value, best position, and 
# best value for each iteration.
def save_logs(swarm,
              method,
              function_name,
              experiment_id
              ) -> None:

    header = ["experiment_id",
              "iteration",
              "particle_id",
              "position",
              "value",
              "best_position",
              "best_value",
              "method",
              "dimensions",
              "function"]

    # Create file and write header only once
    try:
        with open(config["LOGS_FILE_NAME"], 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    except FileExistsError:
        pass

    # Append logs
    with open(config["LOGS_FILE_NAME"], 'a', newline='') as f:

        writer = csv.writer(f)

        for particle_id, particle in enumerate(swarm.particles):

            for iteration, log in enumerate(particle.logs):

                position, value, best_position, best_value = log

                row = [
                    experiment_id,
                    iteration + 1,
                    particle_id,

                    # Convert numpy arrays into JSON strings
                    json.dumps(
                        np.round(
                            position,
                            config["NUM_DECIMALS"]
                        ).tolist()
                    ),

                    round(value, config["NUM_DECIMALS"]),

                    json.dumps(
                        np.round(
                            best_position,
                            config["NUM_DECIMALS"]
                        ).tolist()
                    ),

                    round(best_value, config["NUM_DECIMALS"]),

                    method,
                    swarm.num_dimensions,
                    function_name
                ]

                writer.writerow(row)

# The clean_logs method checks if the logs file exists and removes it to ensure a 
# clean environment for new experiments.
def clean_logs() -> None:
    if os.path.exists(config["LOGS_FILE_NAME"]):
        os.remove(config["LOGS_FILE_NAME"])