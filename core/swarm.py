#----------------------------------------------------------------------------------
# Class representing the PSO algorithm.
# @author: David Ortega Lozano
# @date: 2026-02-27
# @version: 0.4
# @description: It initializes a swarm of particles and iteratively updates their 
# positions and velocities to find the optimal solution to the given objective 
# function.
#----------------------------------------------------------------------------------

from core.particle import Particle
from parallel.v0_pso_sequential import secuential
from parallel.v1_pso_threading import threading_function
from objectives.sphere import sphere_function
from objectives.rosenbrock import rosenbrock_function
from objectives.rastrigin import rastrigin_function
from objectives.ackley import ackley_function
from io_utiles.methods import save_logs
import numpy as np
import json

with open('config.json') as f:
    config = json.load(f)

class Swarm:
    # The Swarm class takes the number of particles, the number of dimensions, and 
    # the objective function as input parameters.
    def __init__(self, 
                 num_particles, 
                 objective_function, 
                 num_dimensions = 2
                 ) -> None:
        
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.objective_function = objective_function
        self.particles = [Particle(self.random_num(-100, 100), 
                                   self.random_num(-1, 1)) 
                                   for _ in range(num_particles)]
        self.global_best_position, self.global_best_value = self.get_first_global_best()
    
    # The random_num method generate random initial positions and velocities for 
    # the particles within specified ranges.
    def random_num(self, 
                   min, 
                   max
                   ) -> np.ndarray:
        
        return np.random.uniform(min, max, self.num_dimensions)
    
    # The optimize method runs the main loop of the PSO algorithm, where it 
    # evaluates the objective function for each particle, updates their personal 
    # bests and the global best, and then updates their velocities and positions 
    # accordingly.
    def optimize(self,
                 method = 1
                 ) -> None:
        
        objective_value = 1 if self.objective_function == 3 else 0

        for iteration in range(config["ITERATIONS"]):
            match method:
                case 1:
                    self.global_best_position, self.global_best_value = secuential(self.particles, 
                                                                                   self.objective_function, 
                                                                                   self.global_best_position, 
                                                                                   self.global_best_value)
                case 2:
                    self.global_best_position, self.global_best_value = threading_function(self.particles, 
                                                                                           self.objective_function, 
                                                                                           self.global_best_position, 
                                                                                           self.global_best_value)
            
            if (self.global_best_value <= (config["ERROR"] + objective_value)
                and 
                self.global_best_value >= (-config["ERROR"]) + objective_value):
                break

        save_logs(self, method)

    # The get_first_global_best method initializes the global best position and 
    # value by evaluating the objective function for the initial positions of all 
    # particles and selecting the best one as the initial global best.
    def get_first_global_best(self) -> tuple:
        bestP = self.particles[0].position
        match self.objective_function:
            case 1:
                bestV = sphere_function(bestP)
            case 2:
                bestV = rosenbrock_function(bestP)
            case 3:
                bestV = rastrigin_function(bestP)
            case 4:
                bestV = ackley_function(bestP)

        for particle in self.particles:
            match self.objective_function:
                case 1:
                    value = sphere_function(particle.position)
                case 2:
                    value = rosenbrock_function(particle.position)
                case 3:
                    value = rastrigin_function(particle.position)
                case 4:
                    value = ackley_function(particle.position)

            if value < bestV:
                bestP = particle.position
                bestV = value

        return bestP, bestV
    