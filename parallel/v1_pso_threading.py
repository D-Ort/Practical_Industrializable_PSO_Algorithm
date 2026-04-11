#----------------------------------------------------------------------------------
# PSO Threading
# @author: David Ortega Lozano
# @date: 2026-03-04
# @version: 1.1
# @description: This code implements a threaded version of the Particle Swarm 
# Optimization (PSO).
#----------------------------------------------------------------------------------

from core.swarm import Swarm
from core.particle import Particle
from objectives.sphere import sphere_function
from objectives.rosenbrock import rosenbrock_function
from objectives.rastrigin import rastrigin_function
from objectives.ackley import ackley_function
from io_utiles.methods import save_logs
import threading
import json

with open('config.json') as f:
    config = json.load(f)

# The Threading class inherits from the Swarm class and implements the optimize method 
# to run the PSO algorithm in a threaded manner.
class Threading(Swarm):

    # The optimize method runs the main loop of the PSO algorithm, where it creates a 
    # thread for each particle to update their velocity and position, evaluate the 
    # objective function, and update their personal bests. After all threads have 
    # completed, it updates the global best based on the personal bests of the 
    # particles. The loop continues until the stopping criteria are met.
    def optimize(self) -> None:
        
        objective_value = 1 if self.objective_function == 3 else 0

        for iteration in range(config["ITERATIONS"]):
            
            threads = []

            for particle in self.particles:
                t = threading.Thread(target = self.thread, args = (particle,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            for particle in self.particles:
                if particle.best_value < self.global_best_value:
                    self.global_best_value = particle.best_value
                    self.global_best_position = particle.best_position.copy()
                    
            if (self.global_best_value <= (config["ERROR"] + objective_value)
                and 
                self.global_best_value >= (-config["ERROR"]) + objective_value):
                break

        save_logs(self, "Threading")

    # The thread method is the target function for each thread, which updates the 
    # velocity and position of the particle, evaluates the objective function, and 
    # updates the personal best of the particle. It also saves the log for the 
    # particle's current state.
    def thread(self, particle) -> None:
        
        particle.update_velocity(self.global_best_position.copy())
        particle.update_position()

        match self.objective_function:
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
                           self.global_best_position, 
                           self.global_best_value)