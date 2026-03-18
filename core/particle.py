#----------------------------------------------------------------------------------
# Class representing a particle in the PSO algorithm
# @author: David Ortega Lozano
# @date: 2026-02-25
# @version: 0.2
# @description: Each particle has a position, velocity, personal best position, and
# personal best value.
#  The particle can update its velocity based on the global best position and its
# own best position, and it can update its position based on its velocity.
#----------------------------------------------------------------------------------

import numpy as np

MAX = 100
MIN = -100

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_value = float('inf')
    
    def update_velocity(self, global_best_position, inertia_weight=0.5, cognitive_weight=1.0, social_weight=1.0):
        r1 = np.random.random()
        r2 = np.random.random()
        for i in range(len(self.position)):
            cognitive_component = cognitive_weight * r1 * (self.best_position[i] - self.position[i])
            social_component = social_weight * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = inertia_weight * self.velocity[i] + cognitive_component + social_component

    def update_position(self):
        self.position += self.velocity

        mask_min = self.position < MIN
        mask_max = self.position > MAX

        # Rebote de velocidad
        self.velocity[mask_min] = - (self.position[mask_min] - MIN)
        self.velocity[mask_max] = - (self.position[mask_max] - MAX)

        # Limitar posición
        self.position = np.clip(self.position, MIN, MAX)