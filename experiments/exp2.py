#----------------------------------------------------------------------------------
# Experiment 2: Grid Search for PSO Hyperparameters
# @author: David Ortega Lozano
# @date: 2026-05-12
# @version: 0.2
# @description: This code runs a grid search to find the optimal hyperparameters 
# for the PSO algorithm.
#----------------------------------------------------------------------------------

import time
from parallel.v0_pso_sequential import Secuential
from experiments.exp1 import init_table, register_results
import json

with open('config.json') as config_file:
    config = json.load(config_file)

# The main function runs a grid search over the specified hyperparameters for the
# PSO algorithm. It iterates through all combinations of the inertia weight,
# cognitive weight, and social weight, and for each combination, it runs the PSO
# algorithm using the sequentialimplementations. The results are stored in a table
# and printed at the end of the grid search.
def grid_search(dimensions, 
                function_choice,
                w_values = config["W_VALUES"],
                c1_values = config["C1_VALUES"],
                c2_values = config["C2_VALUES"]
                ) -> None:

    # Initialize the table of results and the seeds for reproducibility
    table = init_table(dimensions)
    p_seeds = [config["RANDOM_SEED"] + i for i in range(config["PARTICLES"])]

    # Initialize a list to store the results of the grid search for later analysis
    # and comparison.
    count = 0
    results = []

    # Iterate through all combinations of hyperparameters (w, c1, c2) and run the 
    # PSO algorithm for each combination. The results are stored in a table and 
    # printed at the end of the grid search.
    for W in w_values:
        for C1 in c1_values:
            for C2 in c2_values:
                print(f"Running grid search with w={W}, c1={C1}, c2={C2}...")

                # Create the swarm of particles with the current hyperparameters.
                start=time.time()
                swarm = Secuential(config["PARTICLES"],
                                   function_choice,
                                   p_seeds,
                                   dimensions,
                                   exp_id=count,
                                   w=W,
                                   c1=C1,
                                   c2=C2)
                        

                # Optimize the objective function using the PSO algorithm
                swarm.optimize()
                end = time.time()
                execution_time = end - start

                # Store the results of the current hyperparameter combination in 
                # the results list.
                results.append({
                    "w": W,
                    "c1": C1,
                    "c2": C2,
                    "best_position": swarm.global_best_position,
                    "best_value": swarm.global_best_value,
                    "execution_time": execution_time,
                    "id": count
                })

                # Register results in the table
                register_results(table, 
                                 f"{config["OBJ_FUNC"][function_choice-1]}_w{W}_c1{C1}_c2{C2}", 
                                 swarm.global_best_position,
                                 swarm.global_best_value,
                                 execution_time,
                                 dimensions)

                count += 1

    print(table)

    # Sort the results by best value and print the top 5 combinations of 
    # hyperparameters that yielded the best results. The best hyperparameters are 
    # also saved back to the config file for future reference and use in subsequent
    # experiments.
    results.sort(key=lambda x: x["best_value"])
    print("\nTop 5 results:")
    table = init_table(dimensions)

    for result in results[:5]:
        string = f"w({result['w']})_c1({result['c1']})_c2({result['c2']})"
        table = register_results(table, 
                                 string,
                                 result["best_position"],
                                 result["best_value"],
                                 result["execution_time"],
                                 dimensions)
        
    print(table)

    config["w"] = results[0]["w"]
    config["c1"] = results[0]["c1"]
    config["c2"] = results[0]["c2"]

    with open('config.json', 'w') as config_file:
        json.dump(config, config_file, indent=4)

    # The experiments ids of the top 5 are returned for later analysis.
    return [result["id"] for result in results[:5]]