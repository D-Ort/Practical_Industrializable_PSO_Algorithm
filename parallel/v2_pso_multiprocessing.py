#----------------------------------------------------------------------------------
# PSO Multiprocess
# @author: David Ortega Lozano
# @date: 2026-04-28
# @version: 0.1
# @description: This code implements a multiprocess version of the Particle Swarm 
# Optimization (PSO) algorithm.
#----------------------------------------------------------------------------------

from core.swarm import Swarm
from io_utiles.methods import save_logs
import multiprocessing
import json

with open('config.json') as f:
    config = json.load(f)

# The Multiprocess class inherits from the Swarm class and implements the optimize 
# method to run the PSO algorithm in a multiprocess manner.
class Multiprocess(Swarm):

    # The optimize method runs the main loop of the PSO algorithm, where it creates
    # a process for each particle to update their velocity and position, evaluate 
    # the objective function, and update their personal bests. After all processes 
    # have completed, it updates the global best based on the personal bests of the 
    # particles. The loop continues until the stopping criteria are met.
    def optimize(self) -> None:

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:

            for iteration in range(self.num_iterations):

                self.particles = pool.starmap(
                    self.pso_algorithm,
                    [(particle,
                      self.global_best_position.copy(),
                      self.global_best_value,
                      self.objective_function
                    )for particle in self.particles]
                )

                for particle in self.particles:

                    if particle.best_value < self.global_best_value:

                        self.global_best_value = particle.best_value
                        self.global_best_position = (
                            particle.best_position.copy()
                        )

                if (self.global_best_value <= (config["ERROR"])
                    and 
                    self.global_best_value >= (-config["ERROR"])):
                    break

        save_logs(self, config["METHODS"][2])