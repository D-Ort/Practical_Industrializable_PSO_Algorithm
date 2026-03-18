from core.particle import Particle
from core.swarm import Swarm
import time
from prettytable import PrettyTable

def sphere(dimensions):

    # Initialize the table of results
    table = PrettyTable()
    table.field_names = ["Method", "Best Position", "Best Value", "Execution Time (s)"]

    # Define the objective function to minimize
    function_choice = 1

    # Create a swarm of particles
    num_particles = 200
    num_dimensions = dimensions
    seq_swarm = Swarm(num_particles, function_choice, num_dimensions)

    # Optimize the objective function
    max_iterations = 200
    method = 1

    start = time.time() 
    seq_swarm.optimize(max_iterations, method)
    end = time.time()
    seq_time = end - start

    # Include reults in the table
    table.add_row(["Sequential", seq_swarm.global_best_position, seq_swarm.global_best_value, seq_time])

    thread_swarm = Swarm(num_particles, function_choice, num_dimensions)
    method = 2
    start = time.time()
    thread_swarm.optimize(max_iterations, method)
    end = time.time()
    thread_time = end - start

    # Print the best solution found
    table.add_row(["Threading", thread_swarm.global_best_position, thread_swarm.global_best_value, thread_time])

    # Print final results
    print(table)
    