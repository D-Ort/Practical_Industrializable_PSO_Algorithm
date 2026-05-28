#----------------------------------------------------------------------------------
# Distance plot function for PSO results.
# @author: David Ortega Lozano
# @date: 2026-04-29
# @version: 1.0
# @description: Function for plotting the convergence of distance of PSO results.
#----------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import json

with open('config.json') as config_file:
    config = json.load(config_file)

# The plot_distance function creates a 2D plot of the average distance and global 
# best positions distance over iterations for each method. It also saves the plot 
# to a file specified in the configuration.
def plot_distance(plot_data, fileName = config["PLOT_DISTANCE_FILE_NAME"]) -> None:

    plt.figure(figsize=(config["FIG_SIZE"]))

    for names, avg_distances, best_distances in zip(plot_data["name"],
                                                    plot_data["avg_distance"],
                                                    plot_data["best_distance"]):

        iteration = np.arange(len(avg_distances))

        plt.plot(
            iteration,
            avg_distances,
            label=f"{names} avg distance"
        )

        plt.plot(
            iteration,
            best_distances,
            linestyle="--",
            label=f"{names} best distance"
        )

    plt.xlabel("Iteration")
    plt.ylabel("Position value")
    plt.title("Average position vs best position")
    plt.legend()
    plt.grid()

    plt.savefig(fileName, dpi=150, bbox_inches="tight")
    plt.close()
    plt.show()