#----------------------------------------------------------------------------------
# Value plot function for PSO results.
# @author: David Ortega Lozano
# @date: 2026-04-29
# @version: 0.1
# @description: Function for plotting the convergence of values of PSO results.
#----------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import json

with open('config.json') as config_file:
    config = json.load(config_file)

# The plot_value_2D function creates a 2D plot of the average values and best 
# values over iterations for each method. It also saves the plot to a file 
# specified in the configuration.
def plot_value_2D(plot_data, fileName = config["PLOT_VALUES_FILE_NAME"]) -> None:
    
    plt.figure(figsize=(config["FIG_SIZE"]))

    for names, avg_values, best_values in zip(plot_data["name"],
                                              plot_data["avg_value"],
                                              plot_data["best_value"]):

        iteration = np.arange(len(avg_values))

        plt.plot(
            iteration,
            avg_values,
            label=f"{names} avg value"
        )

        plt.plot(
            iteration,
            best_values,
            linestyle="--",
            label=f"{names} best value"
        )

    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Average value vs best value")
    plt.legend()
    plt.grid()

    plt.savefig(fileName, dpi=150, bbox_inches="tight")
    plt.close()
    plt.show()