#----------------------------------------------------------------------------------
# File in charge of visualizing the results of the PSO algorithm.
# @author: David Ortega Lozano
# @date: 2026-03-18
# @version: 2.0
# @description: This file contains the main function for visualizing the results of
# the PSO algorithm, including loading the data from the logs, extracting the 
# relevant information for each experiment, function, and method, and then calling 
# the respective plotting functions to create visualizations of the results.
#----------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import json
from viz.plot_distance import plot_distance
from viz.plot_position import plot_pos_3d
from viz.plot_value import plot_value_2D
from viz.plot_trajectory import plot_trajectory_2D

with open('config.json') as config_file:
    config = json.load(config_file)
    

# The load_data function reads the PSO results from a CSV file and organizes the 
# data into a structured format that allows for easy access and manipulation in the
# subsequent plotting functions. It groups the data by experiment ID, function, 
# method, particle ID, and iteration, and extracts the relevant information such as
# positions, values, best positions, and best values for each particle and 
# iteration.
def load_data(file_name) -> dict:

    df = pd.read_csv(file_name)

    # Recover arrays from JSON strings
    df["position"] = df["position"].apply(json.loads)
    df["best_position"] = df["best_position"].apply(json.loads)

    data = {}

    # Iterate through each experiment and function to extract the relevant data for
    # plotting. The extracted data is then organized in a structured format that 
    # allows for easy access and manipulation in the subsequent plotting functions.
    for exp_id, df_exp in df.groupby("experiment_id"):

        experiment_data = {}

        for function, df_function in df_exp.groupby("function"):

            function_data = {}

            for method, df_method in df_function.groupby("method"):

                method_data = {}

                for particle, df_particle in df_method.groupby("particle_id"):

                    particle_data = {}

                    for iteration, row in df_particle.groupby("iteration"):

                        row = row.iloc[0]

                        particle_data[iteration] = {
                            "position": row["position"],
                            "value": row["value"],
                            "best_position": row["best_position"],
                            "best_value": row["best_value"]
                        }

                    method_data[particle] = particle_data

                function_data[method] = method_data

            experiment_data[function] = function_data

        experiment_data["dimensions"] = df_exp["dimensions"].iloc[0]

        data[exp_id] = experiment_data

    return data

# The plot function is the main function that orchestrates the entire process of 
# loading the data, extracting the relevant information for each experiment, 
# function, and method, and then calling the respective plotting functions to 
# create visualizations of the results.
def plot() -> None:

    # Get data organized from load_data function.
    experiments = load_data(config["LOGS_FILE_NAME"])

    # Iterate through each experiment and function to extract the relevant data for
    # plotting, including average positions, best positions, average values, and 
    # best values for each method. The extracted data is then passed to the 
    # respective plotting functions to create visualizations of the results.
    for exp_id, exp_data in experiments.items():
        dimensions = exp_data["dimensions"]

        for function, function_data in exp_data.items():
            if function == "dimensions":
                continue

            # Initialize data structures for plotting
            graph1_data = {
                "name": [],
                "avg_distance": [],
                "best_distance": []
            }

            graph2_data = {
                "name": [],
                "avg_value": [],
                "best_value": []
            }

            graph3_data = {
                "name": [],
                "avg_positions": [],
                "best_positions": []
            }

            graph4_data = {
                "name": [],
                "particle_positions": [],
                "best_positions": []
            }

            # Iterate through each method to extract and compute the necessary data
            # for plotting, including average positions, best positions, average 
            # values, and best values for each method. The data is organized in a 
            # way that allows for easy plotting of the results using the respective
            # plotting functions.
            for method, method_data in function_data.items():

                avg_distance = []
                best_distance = []

                best_values = []
                best_positions = []

                particle_positions = []

                positions_by_iteration = None
                values_by_iteration = None

                for particle_id, particle_data in method_data.items():

                    positions = []
                    values = []

                    # Initialize positions_by_iteration and values_by_iteration on
                    # the first particle
                    if positions_by_iteration is None:

                        num_iterations = len(particle_data)

                        positions_by_iteration = [
                            [] for _ in range(num_iterations)
                        ]

                        values_by_iteration = [
                            [] for _ in range(num_iterations)
                        ]

                    # Iterate through each iteration for the current particle to 
                    # extract the position and value data, and organize it by 
                    # iteration for computing averages and distances later on. The
                    # best positions and best values are also extracted for the 
                    # first particle to be used in the distance calculations and 
                    # plotting.
                    for iteration, iteration_data in particle_data.items():

                        position = np.array(iteration_data["position"])
                        value = iteration_data["value"]

                        positions.append(position)
                        values.append(value)

                        idx = iteration - 1

                        positions_by_iteration[idx].append(position)
                        values_by_iteration[idx].append(value)

                        # Only extract best positions and values from the first 
                        # particle, as they are the same for all particles in the 
                        # same iteration.
                        if int(particle_id) == 1:
                            best_positions.append(
                                np.array(iteration_data["best_position"])
                            )

                            best_values.append(
                                iteration_data["best_value"]
                            )

                    particle_positions.append(positions)

                # Compute average positions and values for each iteration by taking
                # the mean of the positions and values across all particles for 
                # each iteration.
                avg_positions = [
                    np.mean(pos_list, axis=0)
                    for pos_list in positions_by_iteration
                ]

                avg_values = [
                    np.mean(val_list)
                    for val_list in values_by_iteration
                ]

                # Compute distances from the average positions and best positions 
                # to the origin (or any reference point) for each iteration, which
                # can be used to analyze the convergence of the particles towards
                # the optimal solution over iterations. The distances are 
                # calculated using the L2 norm (Euclidean distance) for both the 
                # average positions and the best positions.
                avg_distance = [
                    np.linalg.norm(pos)
                    for pos in avg_positions
                ]

                best_distance = [
                    np.linalg.norm(pos)
                    for pos in best_positions
                ]

                # Append the extracted and computed data to the respective graph 
                # data structures for plotting. The data is organized in a way that
                # allows for easy plotting of the results using the respective 
                # plotting functions, with the method name, average distances, best
                # distances, average values, best values, average positions, best 
                # positions, and particle positions all stored in a structured 
                # format for each method and function. This allows for clear and 
                # informative visualizations of the PSO algorithm's performance 
                # across different methods and functions.
                graph1_data["name"].append(
                    f"{method}_{function}_{dimensions}D_{exp_id}"
                )

                graph1_data["avg_distance"].append(avg_distance)
                graph1_data["best_distance"].append(best_distance)

                graph2_data["name"].append(
                    f"{method}_{function}_{dimensions}D_{exp_id}"
                )

                graph2_data["avg_value"].append(avg_values)
                graph2_data["best_value"].append(best_values)

                if dimensions == 2:

                    graph3_data["name"].append(
                        f"{method}_{function}_{dimensions}D_{exp_id}"
                    )

                    graph3_data["avg_positions"].append(avg_positions)
                    graph3_data["best_positions"].append(best_positions)

                    graph4_data["name"].append(
                        f"{method}_{function}_{dimensions}D"
                    )

                    graph4_data["particle_positions"].append(
                        particle_positions
                    )

                    graph4_data["best_positions"].append(
                        best_positions
                    )

            # Call the respective plotting functions to create visualizations of 
            # the results using the extracted and computed data. The plots are 
            # saved to files with names that include the method, function, 
            # dimensions, and experiment ID for easy identification and 
            # organization of the results.
            plot_distance(
                graph1_data,
                f"viz/{exp_id}_{function}_{config['PLOT_DISTANCE_FILE_NAME']}"
            )

            plot_value_2D(
                graph2_data,
                f"viz/{exp_id}_{function}_{config['PLOT_VALUES_FILE_NAME']}"
            )

            if dimensions == 2:

                plot_pos_3d(
                    graph3_data,
                    f"viz/{exp_id}_{function}_{config['PLOT_POSITIONS_FILE_NAME']}"
                )

                plot_trajectory_2D(
                    graph4_data,
                    f"viz/{exp_id}_{function}_{config['PLOT_TRAJECTORY_FILE_NAME']}"
                )