#----------------------------------------------------------------------------------
# Ackley function objective function for PSO optimization
# @author: David Ortega Lozano
# @date: 2026-02-24
# @version: 0.2
# @description: The Ackley function is a non-convex function used as a 
#performance test problem for optimization algorithms. It is defined as:
# f(ni) = -20*exp(-0.2*sqrt(1/n*sum(xi^2))) -exp(1/n*sum(cos(2*pi*xi))) +20+e
# The global minimum is at (0, 0, ..., 0) with a function value of 0.
#----------------------------------------------------------------------------------

import math
import numpy as np

def ackley_function(positions):
    n = len(positions)
    sum1 = np.sum([x**2 for x in positions])
    sum2 = np.sum([np.cos(2 * math.pi * x) for x in positions])
    term1 = -20 * math.exp(-0.2 * math.sqrt(sum1 / n))
    term2 = -math.exp(sum2 / n)
    return term1 + term2 + 20 + math.e

