#----------------------------------------------------------------------------------
# Sphere function (sum of squares) objective function for PSO optimization
# @author: David Ortega Lozano
# @date: 2026-02-11
# @version: 1.0
# @description: This code defines the sphere function, which is a common benchmark 
# objective function used in optimization problems. 
#----------------------------------------------------------------------------------

import numpy as np

# The sphere function is defined as the sum of the squares of the input variables. 
# It is a simple convex function that has its global minimum at the origin (0, 0, 
# ..., 0) where the function value is 0.
def sphere_function(position) -> float:
    return np.sum([x**2 for x in position])
