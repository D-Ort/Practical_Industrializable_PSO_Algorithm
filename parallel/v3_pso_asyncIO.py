#----------------------------------------------------------------------------------
# PSO AsyncIO
# @author: David Ortega Lozano
# @date: 2026-05-22
# @version: 1.0
# @description: This code implements an asynchronous version of the Particle Swarm 
# Optimization (PSO) algorithm.
#----------------------------------------------------------------------------------

from core.swarm import Swarm
from io_utiles.methods import save_logs
import asyncio
import nest_asyncio
import json

with open('config.json') as f:
    config = json.load(f)

# The AsyncIO class inherits from the Swarm class and implements the optimize 
# method to run the PSO algorithm in an asynchronous manner.
class AsyncIO(Swarm):

    # The particle_task method is an asynchronous function that updates the 
    # velocity and position of a particle.
    async def particle_task(self, particle):

        # Yield control to the event loop to allow other tasks to run concurrently
        await asyncio.sleep(0)
        self.pso_algorithm(particle)

    # The optimize_async method runs the main loop of the PSO algorithm, where it 
    # creates asynchronous tasks for each particle. After all tasks have 
    # completed, it updates the global best based on the personal bests of the 
    # particles. The loop continues until the stopping criteria are met.
    async def optimize_async(self) -> None:

        for iteration in range(self.num_iterations):

            tasks = []

            for particle in self.particles:

                task = asyncio.create_task(
                    self.particle_task(particle)
                )

                tasks.append(task)

            await asyncio.gather(*tasks)

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

        save_logs(self,
                  config["METHODS"][3],
                  config["OBJ_FUNC"][self.objective_function - 1],
                  self.exp_id)

    # The optimize method is the entry point for running the PSO algorithm, which 
    # calls the optimize_async method to execute the asynchronous tasks. It checks
    # if there is an existing event loop and runs the optimization accordingly, to
    # skip an error with the Jupiter Notebook.
    def optimize(self):

        try:
            loop = asyncio.get_running_loop()

            nest_asyncio.apply()

            loop.run_until_complete(self.optimize_async())

        except RuntimeError:

            asyncio.run(self.optimize_async())