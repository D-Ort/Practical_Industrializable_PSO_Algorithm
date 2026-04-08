#----------------------------------------------------------------------------------
# Main file for running the PSO algorithm and experiments
# @author: David Ortega Lozano
# @date: 2026-03-04
# @version: 1.0
# @description: This is the main entry point for running the Particle Swarm 
# Optimization (PSO) algorithm. It allows the user to select an experiment, choose 
# an objective function, set the number of particles, dimensions, iterations, and 
# the optimization method (sequential or threading). It also includes an option to 
# run a predefined experiment with 200 particles and 200 iterations.
#----------------------------------------------------------------------------------

from experiments.exp0 import manual_run
from experiments.exp1 import compare
from io_utiles.methods import clean_logs
from viz.charts import plot

# The ask_what_to_do function prompts the user to select an experiment to run and 
# returns the choice.
def ask_what_to_do() -> int:
    while True:
        print("Select the experiment to run:")
        print("1. Run the PSO algorithm with user input")
        print("2. Run the experiment with 200 p and 200 i")
        choice = input("Enter the number of the experiment: ")
        if choice in ['1', '2']:
            return int(choice)
        else:
            print("Invalid choice. Please enter 1 or 2.")

# The ask_dimensions function prompts the user to select the number of dimensions 
# for the optimization problem and returns the choice.
def ask_dimensions() -> int:
    while True:
        print("Select the number of dimensions:")
        print("1. 2 dimensions")
        print("2. 3 dimensions")
        print("3. 10 dimensions")
        print("4. 30 dimensions")
        choice = input("Enter the number of dimensions: ")
        if choice in ['1', '2', '3', '4']:
            match choice:
                case '1': dimensions = 2
                case '2': dimensions = 3
                case '3': dimensions = 10
                case '4': dimensions = 30
            return int(dimensions)
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

# The ask_function function prompts the user to select an objective function to
# minimize and returns the choice.
def ask_function() -> int:
    while True:
        print("Select the objective function to minimize:")
        print("1. Sphere Function")
        print("2. Rastrigin Function")
        print("3. Rosenbrock Function")
        print("4. Ackley Function")
        choice = input("Enter the number of the function: ")
        if choice in ['1', '2', '3', '4']:
            return int(choice)
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

# The main block of the code first cleans the environment by removing the logs file
# if it exists. Then it prompts the user to select an experiment to run and 
# executes the corresponding function. Finally, it generates the plots for the 
# results.
if __name__ == "__main__":

    # Before starting, the environment is cleaned by removing the logs file if it 
    # exists
    clean_logs()
        
    # The user is prompted to select an experiment to run and the corresponding 
    # function
    if(ask_what_to_do() == 1):
        manual_run(ask_function())
    else:
        compare(ask_dimensions(), 
                ask_function(), 
                [1, 2])

    # Finally, the plots for the results are generated and saved in the viz folder
    plot()