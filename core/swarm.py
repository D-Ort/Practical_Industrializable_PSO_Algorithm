#----------------------------------------------------------------------------------
# Class representing the PSO algorithm.
# @author: David Ortega Lozano
# @date: 2026-02-27
# @version: 1.1
# @description: It initializes a swarm of particles.
#----------------------------------------------------------------------------------

from core.particle import Particle
from objectives.sphere import sphere_function
from objectives.rosenbrock import rosenbrock_function
from objectives.rastrigin import rastrigin_function
from objectives.ackley import ackley_function
from io_utiles.methods import save_logs
import numpy as np
import json

with open('config.json') as f:
    config = json.load(f)

# The Swarm class represents the swarm of particles in the PSO algorithm. It 
# initializes the particles with random positions and velocities, and it keeps track 
# of the global best position and value found by the swarm. The optimize method is 
# meant to be implemented by subclasses to define the specific optimization strategy 
# (e.g., sequential, threading, multiprocessing).
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
    