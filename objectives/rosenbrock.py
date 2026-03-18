#----------------------------------------------------------------------------------
# Rosenbrock function (Rosenbrock's valley) objective function for PSO optimization
# @author: David Ortega Lozano
# @date: 2026-02-05
# @version: 0.2
# @description: The Rosenbrock function is a non-convex function used as a 
#performance test problem for optimization algorithms. It is defined as:
# f(ni) = sum((a - xi)^2 + b(xi+1 - xi^2)^2)
# The global minimum is at (1, 1) with a function value of 0.
#----------------------------------------------------------------------------------

import numpy as np

def rosenbrock_function(position, a=1, b=100):
    x = position[0:-1]
    y = position[1:]

    parable = np.sum([b * (y[i] - x[i]**2)**2 for i in range(len(x))])
    optimum = np.sum([(a - xi)**2 for xi in x])

    return optimum + parable