from core.particle import Particle
from core.swarm import Swarm

def ask_function():
    while True:
        print("Select the objective function to minimize:")
        print("1. Sphere Function")
        print("2. Rastrigin Function")
        print("3. Rosenbrock Function")
        print("4. Ackley Function")
        choice = input("Enter the number of the function: ")
        if choice in ['1', '2', '3', '4']:
            return int(choice)
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

def ask_value(label):
    choice = input(f"Enter the number of {label}: ")
    return int(choice)

def ask_method():
    while True:
        print("Select the optimization method:")
        print("1. Sequential")
        print("2. Threading")
        choice = input("Enter the number of the method: ")
        if choice in ['1', '2']:
            return int(choice)
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":

    # Define the objective function to minimize
    function_choice = ask_function()

    # Create a swarm of particles
    num_particles = ask_value("particles")
    num_dimensions = ask_value("dimensions")
    swarm = Swarm(num_particles, function_choice, num_dimensions)

    # Optimize the objective function
    max_iterations = ask_value("iterations")
    method = ask_method()

    swarm.optimize(max_iterations, method)

    # Print the best solution found
    print("Best Position:", swarm.global_best_position)
    print("Best Value:", swarm.global_best_value)