# PSO Industrializable y evaluación de paralelismo/concurrencia

This repository contains an academic implementation of the Particle Swarm Optimization (PSO) algorithm developed for the course Parallel Programming.

The project focuses on two main goals:

- Implementing a clean and extensible PSO architecture
- Evaluating parallelization strategies to improve performance

Currently the project includes:

- A sequential PSO implementation
- A threading-based parallel implementation
- Benchmark objective functions
- Logging and visualization of optimization trajectories

Future versions will extend the framework with additional parallel paradigms, benchmarks, and hyperparameter search tools.

---

# What is Particle Swarm Optimization?

Particle Swarm Optimization (PSO) is a population-based optimization algorithm inspired by the collective behavior of bird flocks and fish schools.

Each particle in the swarm represents a candidate solution. During the optimization process, particles update their velocity and position according to:

- Their own best known position
- The best position found by the swarm

The velocity update rule typically follows:

v_i = w · v_i + c1 r1 (pbest_i − x_i) + c2 r2 (gbest − x_i)

Where:

- **w** → inertia weight  
- **c1** → cognitive coefficient  
- **c2** → social coefficient  

In this implementation:

- `w`, `c1`, `c2` default to 0.5
- They are applied inside `particle.update_velocity()`.

---

# Project Structure

```
Practical_Industrializable_PSO_Algorithm
│
├── core
│   ├── particle.py
│   └── swarm.py
│
├── objectives
│   ├── ackley.py
│   ├── rastrigin.py
│   ├── rosenbrock.py
│   └── sphere.py
│
├── parallel
│   ├── v0_pso_sequential.py
│   └── v1_pso_threading.py
│
├── experiments
│   ├── exp0.py
│   └── exp1.py
│
├── io
│   ├── logs.csv
│   └── methods.py
│
├── viz
│   ├── charts.py
│   ├── positions_over_iterations_2D.png
│   └── positions_over_iterations_3D.png
│
├── config.json
└── main.py
```

### Directory description

| Directory | Purpose |
|---|---|
| `core` | Core PSO implementation (particles and swarm logic) |
| `objectives` | Benchmark objective functions |
| `parallel` | Different PSO parallelization strategies |
| `experiments` | Experiment execution scripts |
| `io` | Logging utilities and experiment output |
| `viz` | Visualization utilities and generated charts |

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd Practical_Industrializable_PSO_Algorithm
```

Run the program:

```bash
python main.py
```

Recommended Python version:

```
Python 3.12+
```

---

# Dependencies

The project uses the following Python libraries:

- numpy
- pandas
- matplotlib
- prettytable
- json
- csv
- os
- math
- time
- threading

---

# Running the Algorithm

The main entry point is:

```
main.py
```

When executed, the program displays a menu allowing the user to select the experiment type.

---

## Experiment 0 – Manual configuration

```
Run the PSO algorithm with user input
```

The user selects:

- Objective function
- Number of particles
- Number of dimensions
- Maximum iterations
- Parallel execution method

---

## Experiment 1 – Fixed configuration

```
Run the experiment with 200 particles and 200 iterations
```

The user selects:

- Objective function
- Number of dimensions

The algorithm automatically executes **both parallel methods** and compares them.

Example output:

```
+------------+------------+--------------------+
|   Method   | Best Value | Execution Time (s) |
+------------+------------+--------------------+
| Sequential |   0.0484   |       0.1439       |
| Threading  |   3.0894   |      22.3146       |
+------------+------------+--------------------+
```

---

# Objective Functions

The current implementation includes several standard benchmark functions:

| Function | Description |
|---|---|
| Sphere | Simple convex function |
| Rastrigin | Highly multimodal function |
| Rosenbrock | Non-convex valley-shaped function |
| Ackley | Complex multimodal function |

These functions are widely used to evaluate optimization algorithms.

---

# Parallelization Strategies

Two implementations are currently available.

| Method | Description |
|---|---|
| Sequential | Standard PSO implementation |
| Threading | Parallel particle updates using Python threads |

The goal of the project is to evaluate how different parallel paradigms affect the performance of PSO.

Future versions will include additional approaches.

---

# Logging

During execution, the algorithm stores iteration statistics in:

```
io/logs.csv
```

Each row contains aggregated information for one iteration:

```
iteration,
pos_0,pos_1,...,
value,
best_pos_0,best_pos_1,...,
best_value,
method
```

These logs are used to generate convergence visualizations.

The file is overwritten on each execution.

---

# Visualization

At the end of execution, the program automatically generates plots.

### Position trajectory plots

Saved in:

```
viz/
```

Available visualizations:

| File | Description |
|---|---|
| positions_over_iterations_2D.png | 2D trajectory of swarm mean position |
| positions_over_iterations_3D.png | 3D trajectory comparison |

The plots compare:

- Average swarm position per iteration
- Best global position found

The 3D visualization is generated only for 2D optimization problems.

---

# Commands Summary

| Action | Command |
|---|---|
| Clone repository | `git clone <repo>` |
| Enter project directory | `cd Practical_Industrializable_PSO_Algorithm` |
| Run PSO | `python main.py` |

---

# Reproducibility

The current version does not enforce deterministic execution.

Future versions will include:

- Random seed configuration
- Experiment reproducibility support

---

# Future Work (Roadmap)

Several improvements are planned for future versions of the project.

## Additional parallel strategies

- Multiprocessing implementation
- Asynchronous PSO
- Vectorized PSO using NumPy

## Benchmark suite

A benchmark mode will be added to:

- Evaluate each objective function
- Compare dimension sets `{2, 3, 10, 30}`
- Compare all execution methods

Results will be displayed in formatted tables.

## Hyperparameter Grid Search

A grid search system will be implemented to explore:

- `w`
- `c1`
- `c2`
- number of particles
- number of iterations

Results will be stored in the configuration file for later reuse.

## Real-world optimization problem

Future versions aim to extend the framework to industrial optimization problems.

---

# License

This project was developed for academic purposes as part of the Parallel Programming course.