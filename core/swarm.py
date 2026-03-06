#----------------------------------------------------------------------------------
# Class representing the PSO algorithm.
# @author: David Ortega Lozano
# @date: 2026-02-27
# @version: 0.1
# @description: It initializes a swarm of particles and iteratively updates their 
# positions and velocities to find the optimal solution to the given objective 
# function.
#  The optimize method runs the main loop of the PSO algorithm, where it evaluates
# the objective function for each particle, updates their personal bests and the
# global best, and then updates their velocities and positions accordingly.
#  The Swarm class takes the number of particles, the number of dimensions, and the
# objective function as input parameters. The random_num method generate random 
# initial positions and velocities for the particles within specified ranges.
#----------------------------------------------------------------------------------

from particle import Particle
from parallel.v0_pso_sequential import secuential
from parallel.v1_pso_threading import threading_function
import numpy as np

class Swarm:
    def __init__(self, num_particles, objective_function, num_dimensions = 2, method = 'sequential'):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.objective_function = objective_function
        self.particles = [Particle(self.random_num(-10, 10), self.random_num(-1, 1)) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_value = float('inf')
        if method == 'sequential':
            self.method = 1
        elif method == 'threading':
            self.method = 2
    
    def random_num(self, min, max):
        return [np.random.uniform(min, max, self.num_dimensions)]
    
    def optimize(self, max_iterations):
        for iteration in range(max_iterations):
            match self.method:
                case 1:
                    self.global_best_position, self.global_best_value = secuential(self.particles, self.objective_function, self.global_best_position, self.global_best_value)
                case 2:
                    self.global_best_position, self.global_best_value = threading_function(self.particles, self.objective_function, self.global_best_position, self.global_best_value)

        