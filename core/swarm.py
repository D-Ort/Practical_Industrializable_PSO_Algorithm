#----------------------------------------------------------------------------------
# Class representing the PSO algorithm.
# @author: David Ortega Lozano
# @date: 2026-02-27
# @version: 2.0
# @description: It initializes a swarm of particles.
#----------------------------------------------------------------------------------

from core.particle import Particle
from objectives.sphere import sphere_function
from objectives.rosenbrock import rosenbrock_function
from objectives.rastrigin import rastrigin_function
from objectives.ackley import ackley_function
from objectives.industrializedCase import logistic_regression_objective
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
                 seeds,
                 num_dimensions = config["N_DIMS"][0],
                 num_iterations = config["ITERATIONS"],
                 exp_id = 0,
                 w = config["w"],
                 c1 = config["c1"],
                 c2 = config["c2"]
                 ) -> None:
        
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.objective_function = objective_function
        self.particles = [Particle(num_dimensions,
                                   seed=seeds[i]) 
                                   for i in range(num_particles)]
        self.global_best_position, self.global_best_value = self.get_first_global_best()
        self.num_iterations = num_iterations
        self.exp_id = exp_id
        self.w = w
        self.c1 = c1
        self.c2 = c2

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
            case 5:
                bestV = logistic_regression_objective(bestP)

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
                case 5:
                    value = logistic_regression_objective(particle.position)

            if value < bestV:
                bestP = particle.position
                bestV = value

        return bestP, bestV
    
    # The pso_algorithm method is the target function for each subclass, which 
    # updates the velocity and position of the particle, evaluates the objective 
    # function, and updates the personal best of the particle. It also saves the 
    # log of the particle's current state.
    def pso_algorithm(self, particle,
                      global_best_position = None,
                      global_best_value = None,
                      objective_function = None) -> Particle:
        
        if global_best_position is None:
            global_best_position = self.global_best_position.copy()
        if global_best_value is None:
            global_best_value = self.global_best_value
        if objective_function is None:
            objective_function = self.objective_function

        particle.update_velocity(global_best_position,
                                 inertia_weight=self.w,
                                 cognitive_weight=self.c1,
                                 social_weight=self.c2)
        particle.update_position()

        match objective_function:
            case 1:
                value = sphere_function(particle.position.copy())
            case 2:
                value = rastrigin_function(particle.position.copy())
            case 3:
                value = rosenbrock_function(particle.position.copy())
            case 4:
                value = ackley_function(particle.position.copy())
            case 5:
                value = logistic_regression_objective(particle.position.copy())

        if value < particle.best_value:
            particle.best_value = value
            particle.best_position = particle.position.copy()

        particle.save_log(value, 
                           global_best_position, 
                           global_best_value)
        return particle