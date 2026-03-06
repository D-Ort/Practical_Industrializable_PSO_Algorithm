#----------------------------------------------------------------------------------
# PSO Threading
# @author: David Ortega Lozano
# @date: 2026-03-04
# @version: 0.1
# @description: This code implements a threaded version of the Particle Swarm 
# Optimization (PSO) algorithm.
#----------------------------------------------------------------------------------

import threading
from core.particle import Particle
from objectives.ackley import ackley_function
from objectives.rastrigin import rastrigin_function
from objectives.rosenbrock import rosenbrock_function
from objectives.sphere import sphere_function

def threading_function(particles, objective_function, global_best_position):
    threads = []

    for particle in particles:
        t = threading.Thread(target=thread, args=(particle, objective_function, global_best_position))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for particle in particles:
        if particle.best_value < global_best_value:
            global_best_value = particle.best_value
            global_best_position = particle.best_position

    return global_best_position, global_best_value

def thread(particle, objective_function, global_best_position):
    particle.update_velocity(global_best_position)
    particle.update_position()

    match objective_function:
        case 1:
            value = ackley_function(particle.position)
        case 2:
            value = rastrigin_function(particle.position)
        case 3:
            value = rosenbrock_function(particle.position)
        case 4:
            value = sphere_function(particle.position)

    if value < particle.best_value:
        particle.best_value = value
        particle.best_position = particle.position