# PSO Industrializable y evaluación de paralelismo/concurrencia

## Overview

This repository contains an academic implementation of the Particle Swarm Optimization (PSO) algorithm developed for the Parallel Programming course.

The project focuses on two main goals:

- Implementing a clean and extensible PSO architecture
- Evaluating different concurrency and parallelization strategies to improve performance

Unlike traditional academic PSO implementations focused only on mathematical optimization, this project also explores the practical limitations of Python parallelism, especially regarding:

CPU-bound workloads
GIL limitations
Serialization overhead
Multiprocessing scalability
Hardware constraints in virtualized environments

The framework supports both classical benchmark functions and a simple industrializable use case based on hyperparameter optimization of a machine learning model.

---

## Features
### Implemented PSO Versions

The project currently includes for execution strategies:

| Strategy | Description |
|---|---|
| Sequential | Baseline implementation |
| Threading | Concurrent implementation using Python threads |
| Multiprocessing | Parallel implementation using process pools |
| AsyncIO | Cooperative asynchronous implementation |

---

### Objective Functions

| Function | Description |
|---|---|
| Sphere | Simple convex benchmark function |
| Rastrigin | Highly multimodal benchmark |
| Rosenbrock | Non-convex valley-shaped function |
| Ackley | Complex multimodal benchmark |
| Logistic Regression Hyperparameter Optimization | Industrializable machine learning use case |

---

### Visualization System

The project includes a complete visualization pipeline:

- Convergence plots
- Distance-to-optimum plots
- 3D swarm trajectory plots
- Interactive particle trajectory visualization

---

## What is Particle Swarm Optimization?

Particle Swarm Optimization (PSO) is a population-based optimization algorithm inspired by the collective behavior of bird flocks and fish schools.

Each particle in the swarm represents a candidate solution. During the optimization process, particles update their velocity and position according to:

- Their own best known position
- The best position found by the swarm

The standar velocity update ecuation is:

v_i = w · v_i + c1 · r1 · (pbest_i − x_i) + c2 · r2 · (gbest − x_i)

Where:

| Parameter | Meaning |
|---|---|
| **w** | Inertia weight |
| **c1** | Cognitive coefficient |
| **c2** | Social coefficient |
| r1, r2 | Random factors |

---

## Parallelization Strategies
### Sequential Version

The sequential implementation acts as the baseline reference. Particles are evaluated one after another in a single execution thread.

Advantages:

- Minimal overhead
- Deterministic behavior
- Best performance for lightweight workloads

---

### Threading Version

The threading implementation creates one thread per particle. This strategy explores Python concurrency using the threading module. However, due to Python's Global Interpreter Lock (GIL), CPU-bound workloads cannot execute Python bytecode truly in parallel.

As a result:

- Significant synchronization overhead appears
- Real CPU parallelism is not achieved
- Performance is usually worse than sequential execution

---

### Multiprocessing Version

The multiprocessing implementation uses a process pool to distribute particle evaluations across multiple processes. Unlike threading, multiprocessing bypasses the GIL because each process owns its own Python interpreter. This allows real parallel execution.

However, multiprocessing introduces important costs:

- Object serialization (pickle)
- Inter-process communication (IPC)
- Memory duplication
- Process management overhead

These costs can dominate execution time when particle evaluations are lightweight.

---

### AsyncIO Version

The AsyncIO implementation explores cooperative concurrency using Python's asynchronous event loop. This strategy is primarily educational because PSO is a CPU-bound workload rather than an I/O-bound problem.

AsyncIO demonstrates:

- Cooperative task scheduling
- Non-blocking execution flow
- Event loop orchestration

but does not provide true CPU parallelism.

---

## Experimental Conclusions

The experimental results obtained in this project show that:

- Parallelization does not always improve performance.
- Python threading is heavily limited by the GIL for CPU-bound tasks.
- Multiprocessing may become slower than sequential execution when task granularity is small.
- Serialization overhead can dominate execution time.
- Hardware limitations strongly affect scalability.

In lightweight benchmark functions such as Sphere or Ackley, sequential execution consistently achieves the best performance.

In heavier workloads such as machine learning hyperparameter optimization, multiprocessing becomes more competitive. But overhead may still exceed computational gains depending on hardware constraints.

---

## Industrializable Use Case

The project includes a practical optimization scenario: ***Logistic Regression Hyperparameter Optimization***

PSO is used to optimize hyperparameters of a logistic regression classifier using the Breast Cancer dataset from scikit-learn.

Optimized parameters include:

- Regularization strength (C)
- Maximum iterations
- Tolerance

This demonstrates that the PSO framework can be adapted to real-world optimization problems beyond synthetic mathematical benchmarks.

---

## Decisions

The PSO implementation incorporates two stopping criteria to balance optimization quality and computational efficiency. The first criterion is reaching the maximum number of iterations defined in the configuration file, ensuring that the algorithm always terminates after a bounded amount of computation. The second criterion is an early stopping condition based on the objective function value: the optimization process stops automatically when the global best value falls within the interval `[-0.05, 0.05]`. Since the benchmark functions used in this project have their global optimum at or near zero, this tolerance threshold allows the algorithm to terminate once a sufficiently accurate solution has been found, significantly reducing unnecessary computations.

Regarding the search space boundaries, all particles are constrained within the interval `[-100, 100]` for each dimension. To handle boundary violations, a rebound (bounce) policy is applied. When a particle exceeds the allowed search limits, its velocity component is inverted and its position is corrected back into the valid range. This strategy prevents particles from escaping the search space while preserving part of their momentum, helping maintain swarm diversity and improving exploration stability during the optimization process.

---

## Project Structure

```
Practical_Industrializable_PSO_Algorithm
│
├── core
│   ├── particle.py
│   └── swarm.py
│
├── objectives
│   ├── sphere.py
│   ├── rastrigin.py
│   ├── rosenbrock.py
│   ├── ackley.py
│   └── industrializedCase.py
│
├── parallel
│   ├── v0_pso_sequential.py
│   ├── v1_pso_threading.py
│   ├── v2_pso_multiprocess.py
│   └── v3_pso_asyncIO.py
│
├── experiments
│   ├── exp0.py
│   ├── exp1.py
│   └── exp2.py
│
├── io_utiles
│   ├── logistic_dataset.py
│   └── methods.py
│
├── viz
│   ├── charts.py
│   ├── plot_distance.py
│   ├── plot_position.py
│   ├── plot_trajectory.py
│   └── plot_value.py
│
├── config.json
├── main.py
└── README.md
```

### Directory description

| Directory | Purpose |
|---|---|
| `core` | Core PSO implementation (particles and swarm logic) |
| `objectives` | Benchmark objective functions |
| `parallel` | Different PSO parallelization strategies |
| `experiments` | Experiment execution scripts |
| `io_utiles` | Logging utilities and experiment output |
| `viz` | Visualization utilities and generated charts |

---

### Installation

#### Clone the repository:

```bash
git clone <https://github.com/D-Ort/Practical_Industrializable_PSO_Algorithm>
cd Practical_Industrializable_PSO_Algorithm
```

#### Run the program:

```bash
python main.py
```

#### Recommended Python version:

```
Python 3.12+
```

---

## Dependencies

The project uses the following external dependencies:

- numpy
- pandas
- matplotlib
- prettytable
- scikit-learn
- scipy
- ipympl
- ipywidgets
- pillow
- jupyter
- notebook
- mplcursors
- ipykernel

---

## Running the Algorithm

The main entry point is:

```
main.py
```

When executed, the program displays a menu allowing the user to select the experiment type.

---

### Experiment 0 – Manual configuration

Run the PSO algorithm with user input.

The user selects:

- Objective function
- Number of particles
- Number of dimensions
- Maximum iterations
- Parallel strategy

---

### Experiment 1 – Fixed configuration

Run the experiment with 200 particles and 200 iterations with all available strategies under identical conditions and compare.

The user selects:

- Objective function
- Number of dimensions

---

### Experiment 2 – Grid Search

Performs PSO hyperparameter search for:

- Inertia weight(`w`)
- Cognitive coefficient(`c1`)
- Social coefficient(`c2`)

The best configuration is automatically stored in `config.json`.

---

### Experiment 3 – Industrializable Case

An adaptation of Experiment 1, which uses the PSO algorithm to optimize the hyperparameters of a logistic regression model trained on the breast cancer dataset.

---

## Logging

During execution, the algorithm stores iteration statistics in:

```
io_utiles/logs.csv
```

Each row stores the results for each particle at each iteration:

```
experiment_id,
iteration,
particle_id,
position,
value,
best_position,
best_value,
method,
dimensions,
function
```

These logs are used to generate visualizations. The file is overwritten on each execution.

---

## Visualization Examples

The framework can generate:

- Distance convergence charts
- Value convergence charts
- 3D optimization trajectories
- Interactive swarm evolution plots

Interactive visualizations allow:

- Iteration navigation
- Experiment selection
- Swarm trajectory inspection

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computation |
| Pandas | CSV processing |
| Matplotlib | Visualization |
| Multiprocessing | Parallel execution |
| Threading | Concurrency experiments |
| AsyncIO | Asynchronous execution |
| Scikit-learn | Machine learning experiments |

---

## Author

David Ortega Lozano

---

## License

This project was developed for academic purposes as part of the Parallel Programming course.