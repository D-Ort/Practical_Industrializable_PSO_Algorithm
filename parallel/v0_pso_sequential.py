#----------------------------------------------------------------------------------
# PSO Sequential
# @author: David Ortega Lozano
# @date: 2026-02-25
# @version: 1.1
# @description: This code implements a sequential version of the Particle Swarm 
# Optimization (PSO) algorithm.
#----------------------------------------------------------------------------------

from core.swarm import Swarm
from objectives.sphere import sphere_function
from objectives.rosenbrock import rosenbrock_function
from objectives.rastrigin import rastrigin_function
from objectives.ackley import ackley_function
from io_utiles.methods import save_logs
import json

with open('config.json') as f:
    config = json.load(f)

# The Secuential class inherits from the Swarm class and implements the optimize method 
# to run the PSO algorithm in a sequential manner.
class Secuential(Swarm):

    # The optimize method runs the main loop of the PSO algorithm, where it evaluates 
    # the objective function for each particle, updates their personal bests and the 
    # global best, and then updates their velocities and positions accordingly.
    def optimize(self) -> None:
        
        objective_value = 1 if self.objective_function == 3 else 0

        for iteration in range(config["ITERATIONS"]):
            
            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position()

                match self.objective_function:
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
                    particle.best_position = particle.position.copy()

                    if value < self.global_best_value:
                        self.global_best_value = value
                        self.global_best_position = particle.position.copy()

                particle.save_log(value, 
                                self.global_best_position.copy(), 
                                self.global_best_value)    
                    
            if (self.global_best_value <= (config["ERROR"] + objective_value)
                and 
                self.global_best_value >= (-config["ERROR"]) + objective_value):
                break

        save_logs(self, "Secuential")