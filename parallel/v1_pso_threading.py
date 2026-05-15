#----------------------------------------------------------------------------------
# PSO Threading
# @author: David Ortega Lozano
# @date: 2026-03-04
# @version: 1.2
# @description: This code implements a threaded version of the Particle Swarm 
# Optimization (PSO).
#----------------------------------------------------------------------------------

from core.swarm import Swarm
from io_utiles.methods import save_logs
import threading
import json

with open('config.json') as f:
    config = json.load(f)

# The Threading class inherits from the Swarm class and implements the optimize 
# method to run the PSO algorithm in a threaded manner.
class Threading(Swarm):

    # The optimize method runs the main loop of the PSO algorithm, where it creates
    # a thread for each particle to update their velocity and position, evaluate 
    # the objective function, and update their personal bests. After all threads 
    # have completed, it updates the global best based on the personal bests of the 
    # particles. The loop continues until the stopping criteria are met.
    def optimize(self) -> None:
        
        for iteration in range(self.num_iterations):
            
            threads = []

            for particle in self.particles:
                t = threading.Thread(target = self.pso_algorithm,
                                     args = (particle,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            for particle in self.particles:
                if particle.best_value < self.global_best_value:
                    self.global_best_value = particle.best_value
                    self.global_best_position = particle.best_position.copy()
                    
            if (self.global_best_value <= (config["ERROR"])
                and 
                self.global_best_value >= (-config["ERROR"])):
                break

        save_logs(self, config["METHODS"][1])