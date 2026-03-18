#----------------------------------------------------------------------------------
# PSO Sequential
# @author: David Ortega Lozano
# @date: 2026-02-25
# @version: 0.2
# @description: This code implements a sequential version of the Particle Swarm 
# Optimization (PSO) algorithm.
#----------------------------------------------------------------------------------

from core.particle import Particle
from objectives.ackley import ackley_function
from objectives.rastrigin import rastrigin_function
from objectives.rosenbrock import rosenbrock_function
from objectives.sphere import sphere_function

def secuential(particles, objective_function, global_best_position, global_best_value):        
    
    for particle in particles:
        particle.update_velocity(global_best_position)
        particle.update_position()

        match objective_function:
            case 1:
                value = sphere_function(particle.position)
            case 2:
                value = rastrigin_function(particle.position)
            case 3:
                value = rosenbrock_function(particle.position)
            case 4:
                value = ackley_function(particle.position)

        if value < particle.best_value:
            particle.best_value = value
            particle.best_position = particle.position

            if value < global_best_value:
                global_best_value = value
                global_best_position = particle.position
    
    return global_best_position, global_best_value