#----------------------------------------------------------------------------------
# Main file for running the PSO algorithm and experiments
# @author: David Ortega Lozano
# @date: 2026-03-04
# @version: 0.2
# @description: This is the main entry point for running the Particle Swarm 
# Optimization (PSO) algorithm. It allows the user to select an experiment, choose 
# an objective function, set the number of particles, dimensions, iterations, and 
# the optimization method (sequential or threading). It also includes an option to 
# run a predefined experiment with 200 particles and 200 iterations.
#----------------------------------------------------------------------------------

from core.particle import Particle
from core.swarm import Swarm
from experiments.exp1 import sphere

def ask_what_to_do():
    while True:
        print("Select the experiment to run:")
        print("1. Run the PSO algorithm with user input")
        print("2. Run the experiment with 200 p and 200 i")
        choice = input("Enter the number of the experiment: ")
        if choice in ['1', '2']:
            return int(choice)
        else:
            print("Invalid choice. Please enter 1 or 2.")

def ask_function():
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

def ask_value(label):
    choice = input(f"Enter the number of {label}: ")
    return int(choice)

def ask_method():
    while True:
        print("Select the optimization method:")
        print("1. Sequential")
        print("2. Threading")
        choice = input("Enter the number of the method: ")
        if choice in ['1', '2']:
            return int(choice)
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":

    if(ask_what_to_do() == 1):
        # Define the objective function to minimize
        function_choice = ask_function()

        # Create a swarm of particles
        num_particles = ask_value("particles")
        num_dimensions = ask_value("dimensions")
        swarm = Swarm(num_particles, function_choice, num_dimensions)

        # Optimize the objective function
        max_iterations = ask_value("iterations")
        method = ask_method()

        swarm.optimize(max_iterations, method)

        # Print the best solution found
        print("Best Position:", swarm.global_best_position)
        print("Best Value:", swarm.global_best_value)
    else:
        dimensions = 2
        sphere(dimensions)