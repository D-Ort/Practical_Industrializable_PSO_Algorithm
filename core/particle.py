#----------------------------------------------------------------------------------
# Class representing a particle in the PSO algorithm
# @author: David Ortega Lozano
# @date: 2026-02-05
# @version: 0.1
# @description: Each particle has a position, velocity, personal best position, and
# personal best value.
#  The particle can update its velocity based on the global best position and its
# own best position, and it can update its position based on its velocity.
#----------------------------------------------------------------------------------

import numpy as np

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_value = float('inf')
    
    def update_velocity(self, global_best_position, inertia_weight=0.5, cognitive_weight=1.0, social_weight=1.0):
        r1 = np.random.random()
        r2 = np.random.random()
        cognitive_component = cognitive_weight * r1 * (self.best_position - self.position)
        social_component = social_weight * r2 * (global_best_position - self.position)
        self.velocity = inertia_weight * self.velocity + cognitive_component + social_component
    
    def update_position(self):
        self.position += self.velocity