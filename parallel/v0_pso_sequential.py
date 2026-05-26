#----------------------------------------------------------------------------------
# PSO Sequential
# @author: David Ortega Lozano
# @date: 2026-02-25
# @version: 1.2
# @description: This code implements a sequential version of the Particle Swarm 
# Optimization (PSO) algorithm.
#----------------------------------------------------------------------------------

from core.swarm import Swarm
from io_utiles.methods import save_logs
import json

with open('config.json') as f:
    config = json.load(f)

# The Secuential class inherits from the Swarm class and implements the optimize 
# method to run the PSO algorithm in a sequential manner.
class Secuential(Swarm):

    # The optimize method runs the main loop of the PSO algorithm, where it 
    # evaluates the objective function for each particle, updates their personal 
    # bests and the global best, and then updates their velocities and positions 
    # accordingly.
    def optimize(self) -> None:
        
        for iteration in range(self.num_iterations):
            
            for particle in self.particles:
                self.pso_algorithm(particle)

            for particle in self.particles:
                if particle.best_value < self.global_best_value:
                    self.global_best_value = particle.best_value
                    self.global_best_position = particle.best_position.copy()
                    
            if (self.global_best_value <= (config["ERROR"])
                and 
                self.global_best_value >= (-config["ERROR"])):
                break

        save_logs(self, 
                  config["METHODS"][0],
                  config["OBJ_FUNC"][self.objective_function - 1],
                  self.exp_id)