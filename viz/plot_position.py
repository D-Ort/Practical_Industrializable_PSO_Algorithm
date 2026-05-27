#----------------------------------------------------------------------------------
# 3D position plot function for PSO results.
# @author: David Ortega Lozano
# @date: 2026-04-29
# @version: 0.2
# @description: Function for plotting in 3D the average positions and global best 
# positions over iterations of PSO results.
#----------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json

with open('config.json') as config_file:
    config = json.load(config_file)

# The plot_pos_3d function creates a 3D plot of the average trajectories of the 
# particles and the global best position over iterations for each method. It also 
# saves the plot to a file specified in the configuration. The x and y axes 
# represent the position values in the two dimensions, while the z-axis represents
# the iteration number.
def plot_pos_3d(plot_data, fileName = config["PLOT_POSITIONS_FILE_NAME"]) -> None:

    fig = plt.figure(figsize=(config["FIG_SIZE"]))

    ax = fig.add_subplot(111, projection="3d")

    for name, avg_positions, best_positions in zip(
        plot_data["name"],
        plot_data["avg_positions"],
        plot_data["best_positions"]
    ):

        avg_positions = np.array(avg_positions)
        best_positions = np.array(best_positions)

        iteration = np.arange(len(avg_positions))

        pos_x = avg_positions[:, 0]
        pos_y = avg_positions[:, 1]

        gb_x = best_positions[:, 0]
        gb_y = best_positions[:, 1]

        ax.plot(
            pos_x,
            pos_y,
            iteration,
            label=f"{name} avg trajectory"
        )

        ax.plot(
            gb_x,
            gb_y,
            iteration,
            linestyle="--",
            label=f"{name} best trajectory"
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Iteration")

    ax.set_title(
        "Trajectory of average position and best position"
    )

    ax.legend()

    ax.grid()

    plt.savefig(
        fileName,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()
    plt.show()