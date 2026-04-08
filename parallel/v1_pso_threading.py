#----------------------------------------------------------------------------------
# PSO Threading
# @author: David Ortega Lozano
# @date: 2026-03-04
# @version: 1.0
# @description: This code implements a threaded version of the Particle Swarm 
# Optimization (PSO) algorithm.
#----------------------------------------------------------------------------------

import threading
import json
from objectives.ackley import ackley_function
from objectives.rastrigin import rastrigin_function
from objectives.rosenbrock import rosenbrock_function
from objectives.sphere import sphere_function

with open('config.json') as config_file:
    config = json.load(config_file)

# The threading_function implements the PSO algorithm using threads, where each
# particle's velocity and position are updated in parallel, and the objective
# function is evaluated for each particle to update their personal bests. After
# all threads have completed, the global best is updated based on the personal 
# bests of the particles. The thread function is the target function for each 
# thread,
def threading_function(particles, 
                       objective_function, 
                       global_best_position, 
                       global_best_value
                       ) -> tuple:
    threads = []

    for particle in particles:
        t = threading.Thread(target = thread, 
                             args = (particle, 
                                     objective_function, 
                                     global_best_position.copy(), 
                                     global_best_value))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for particle in particles:
        if particle.best_value < global_best_value:
            global_best_value = particle.best_value
            global_best_position = particle.best_position.copy()

    return global_best_position.copy(), global_best_value

# The thread function is the target function for each thread, where it updates the
# velocity and position of the particle, evaluates the objective function, updates
# the personal best of the particle, and saves the log of the current state. The
# global best is not updated in this function to avoid race conditions, and it is 
# updated after all threads have completed in the threading_function.
def thread(particle, 
           objective_function, 
           global_best_position, 
           global_best_value
           ) -> None:
    
    particle.update_velocity(global_best_position.copy())
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

    if value < particle.best_value:
        particle.best_value = value
        particle.best_position = particle.position.copy()

    particle.save_log(value, 
                      global_best_position, 
                      global_best_value)