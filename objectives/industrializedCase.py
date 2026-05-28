#----------------------------------------------------------------------------------
# Industrialized case objective function for PSO optimization
# @author: David Ortega Lozano
# @date: 2026-05-19
# @version: 1.0
# @description: This code defines the industrialized case objective function for 
# PSO optimization. The objective function is based on the performance of a 
# logistic regression model on the breast cancer dataset.
#----------------------------------------------------------------------------------

import numpy as np
from sklearn.linear_model import LogisticRegression
from io_utiles.logistic_dataset import x_train, y_train, x_test, y_test

# The function takes a position vector as input, which represents the 
# hyperparameters of the logistic regression model, and returns the value of the 
# objective function, which is 1 minus the accuracy of the model on the test set.
# The hyperparameters being optimized are the regularization strength (C), the 
# maximum number of iterations (max_iter), and the tolerance for convergence (tol).
# The function also includes clipping to ensure that the hyperparameters are within
# a reasonable range for the logistic regression model to perform well.
def logistic_regression_objective(position):

    C = np.clip(abs(position[0]), 0.0001, 100)

    max_iter = int(np.clip(abs(position[1]), 50, 1000))

    tol = np.clip(abs(position[2]), 1e-6, 1e-1)

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        tol=tol,
        solver="lbfgs"
    )

    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)

    return 1 - accuracy