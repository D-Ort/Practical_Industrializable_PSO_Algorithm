#----------------------------------------------------------------------------------
# Class representing a particle in the PSO algorithm
# @author: David Ortega Lozano
# @date: 2026-02-25
# @version: 1.0
# @description: Each particle has a position, velocity, personal best position, and
# personal best value.
#  The particle can update its velocity based on the global best position and its
# own best position, and it can update its position based on its velocity.
#----------------------------------------------------------------------------------

import numpy as np
import json

with open('config.json') as f:
    config = json.load(f)

class Particle:
    def __init__(self,
                 num_dimensions,                 
                 seed
                 ) -> None:
        
        self.rng = np.random.default_rng(seed)
        self.position = self.rng.uniform(config["MIN"], config["MAX"], num_dimensions)
        self.velocity = self.rng.uniform(-1, 1, num_dimensions)
        self.best_position = self.position.copy()
        self.best_value = float('inf')
        self.logs = []
    
    # The update_velocity method updates the velocity of the particle based on the 
    # global best position, its own best position, and the inertia weight, 
    # cognitive weight, and social weight parameters.
    def update_velocity(self, 
                        global_best_position, 
                        inertia_weight=0.5, 
                        cognitive_weight=0.5, 
                        social_weight=0.5
                        ) -> None:
        
        r1 = self.rng.uniform(0, 1)
        r2 = self.rng.uniform(0, 1)

        for i in range(len(self.position)):
            cognitive_component = cognitive_weight * r1 * (self.best_position[i] - self.position[i])
            social_component = social_weight * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = inertia_weight * self.velocity[i] + cognitive_component + social_component

    # The update_position method updates the position of the particle based on its 
    # velocity and ensures that it stays within the defined bounds.
    def update_position(self) -> None:
        
        self.position += self.velocity

        mask_min = self.position < config["MIN"]
        mask_max = self.position > config["MAX"]

        # Rebote de velocidad
        self.velocity[mask_min] = - (self.position[mask_min] - config["MIN"])
        self.velocity[mask_max] = - (self.position[mask_max] - config["MAX"])

        # Limitar posición
        self.position = np.clip(self.position, 
                                config["MIN"], 
                                config["MAX"])

    # The save_log method saves the current position, value, global best position,
    # and global best value of the particle in its logs list. This allows for 
    # tracking the history of the particle's performance over time.
    def save_log(self, 
                 value, 
                 global_best_position, 
                 global_best_value
                 ) -> None:
        
        self.logs.append((self.position.copy(), 
                          value, 
                          global_best_position, 
                          global_best_value))