#----------------------------------------------------------------------------------
# File in charge of visualizing the results of the PSO algorithm.
# @author: David Ortega Lozano
# @date: 2026-03-18
# @version: 1.0
# @description: This file contains functions to read the results of the PSO 
# algorithm from a CSV file and create visualizations of the average positions of 
# the particles and the global best position over iterations. It also includes a 
# function to create a 3D plot of the trajectories of the particles and the global 
# best position over iterations.
#----------------------------------------------------------------------------------

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

with open('config.json') as config_file:
    config = json.load(config_file)

# The plot function reads the results of the PSO algorithm from a CSV file, 
# and extract the average positions, global best positions, average values, and 
# best values for each method, and then calls the appropriate plotting functions 
# to create visualizations of the results. It first reads the CSV file into a 
# DataFrame, determines the number of dimensions based on the column names, and 
# then iterates through each unique method to extract the relevant data. The 
# extracted data is stored in dictionaries that are passed to the plotting 
# functions to create the 2D and 3D plots of the positions and values over 
# iterations. Finally, it saves the generated plots to files specified in the 
# configuration.
def plot() -> None:

    df = pd.read_csv(config["LOGS_FILE_NAME"])

    num_columns = len(df.columns)
    dim = (num_columns - 4) // 2

    methods = df["method"].unique()

    plot_data = {}
    value_data = {}

    for m in methods:

        df_m = df[df["method"] == m].sort_values("iteration")

        iterations = df_m["iteration"].values

        pos_avg = df_m[[f"pos_{i}" for i in range(dim)]].values
        best_pos_avg = df_m[[f"best_pos_{i}" for i in range(dim)]].values

        plot_data[m] = {
            "iterations": iterations,
            "pos_avg": pos_avg,
            "best_pos_avg": best_pos_avg
        }
        value_data[m] = {
            "iterations": iterations,
            "value_avg": df_m["value"].values,
            "best_value": df_m["best_value"].values
        }

    plot_pos_2d(plot_data)
    plot_value_2D(value_data)

    if dim == 2:
        plot_pos_3d(plot_data)

# The plot_pos_2d function creates a 2D plot of the average positions and global 
# best positions over iterations for each method. It plots the average position and 
# global best position for each method, labels the axes, adds a title and legend, 
# and saves the plot to a file specified in the configuration.
def plot_pos_2d(plot_data) -> None:

    plt.figure(figsize=(10,6))

    for m, data in plot_data.items():

        iterations = data["iterations"]

        plt.plot(
            iterations,
            np.mean(data["pos_avg"], axis=1),
            label=f"{m} avg pos"
        )

        plt.plot(
            iterations,
            np.mean(data["best_pos_avg"], axis=1),
            linestyle="--",
            label=f"{m} best pos"
        )

    plt.xlabel("Iteration")
    plt.ylabel("Position value")
    plt.title("Average position vs best position")
    plt.legend()
    plt.grid()

    plt.savefig(config["PLOT_POS_2D_FILE_NAME"], dpi=150, bbox_inches="tight")
    plt.close()

# The plot_pos_3d function creates a 3D plot of the trajectories of the particles 
# and the global best position over iterations for each method. It plots the 
# average position and global best position in 3D space, labels the axes, adds a 
# title and legend, and saves the plot to a file specified in the configuration. 
# The x and y axes represent the position values in the two dimensions, while the 
# z-axis represents the iteration number.
def plot_pos_3d(plot_data) -> None:

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection="3d")

    for m, data in plot_data.items():

        t = data["iterations"]

        pos_x = data["pos_avg"][:,0]
        pos_y = data["pos_avg"][:,1]

        gb_x = data["best_pos_avg"][:,0]
        gb_y = data["best_pos_avg"][:,1]

        ax.plot(pos_x, 
                pos_y, 
                t, 
                label=f"{m} avg trajectory")
        ax.plot(gb_x, 
                gb_y, 
                t, 
                linestyle="--", 
                label=f"{m} best trajectory")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Iteration")
    ax.set_title("Trajectory of average position and best position")

    ax.legend()
    ax.grid()

    plt.savefig(config["PLOT_POS_3D_FILE_NAME"], 
                dpi=150, 
                bbox_inches="tight")
    plt.close()

# The plot_value_2D function creates a 2D plot of the average values and best 
# values over iterations for each method. It plots the average value and best 
# value for each method, labels the axes, adds a title and legend, and saves the 
# plot to a file specified in the configuration. The x-axis represents the 
# iteration number, while the y-axis represents the value of the objective 
# function. The average value is plotted with a solid line, while the best value 
# is plotted with a dashed line to differentiate them visually.
def plot_value_2D(value_data) -> None:
    
    plt.figure(figsize=(10,6))

    for m, data in value_data.items():

        iterations = data["iterations"]

        plt.plot(
            iterations,
            data["value_avg"],
            label=f"{m} avg value"
        )

        plt.plot(
            iterations,
            data["best_value"],
            linestyle="--",
            label=f"{m} best value"
        )

    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Average value vs best value")
    plt.legend()
    plt.grid()

    plt.savefig(config["PLOT_VALUES_FILE_NAME"], dpi=150, bbox_inches="tight")
    plt.close()