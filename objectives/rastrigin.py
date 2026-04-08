#----------------------------------------------------------------------------------
# Rastrigin function objective function for PSO optimization
# @author: David Ortega Lozano
# @date: 2026-02-05
# @version: 0.4
# @description: The Rastrigin function is a non-convex function used as a 
#performance test problem for optimization algorithms. 
#----------------------------------------------------------------------------------

import math
import numpy as np

# It is defined as:
# f(x, y) = 20 + x^2 + y^2 - 10(cos(2πx) + cos(2πy))
# The global minimum is at (0, 0) with a function value of 0.
def rastrigin_function(positions, A=10) -> float:
    quadratic = np.sum([x**2 for x in positions])
    cosine = np.sum([A * np.cos(2 * math.pi * x) for x in positions])
    return  (A * len(positions)) + quadratic - cosine