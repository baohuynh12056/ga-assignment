import time
import json
import random
import os
import matplotlib.pyplot as plt

from src.chromosome import Chromosome, Population
from src.operators import TournamentSelection, OnePointCrossover, BitflipMutation
from src.problems import OneMaxProblem, KnapsackProblem, FeatureSelectionProblem
from src.ga import GeneticAlgorithm

# Ensure the reports directory exists at the root of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "reports"))
os.makedirs(REPORT_DIR, exist_ok=True)

def run_experiment(problem_name, problem, pop_size=100, max_gen=300, length=100):
    print(f"--- Solving problem {problem_name} ---")
    
    # Set a fixed seed as required by the assignment for reproducibility
    random.seed(42)
    
    # 1. Initialize the initial population
    initial_chromosomes = [Chromosome(problem.generate_random_genes()) for _ in range(pop_size)]
    population = Population(initial_chromosomes)
    
    # 2. Configure GA
    ga = GeneticAlgorithm(
        selection=TournamentSelection(k=3),
        crossover=OnePointCrossover(probability=0.9),
        mutation=BitflipMutation(probability=1.0 / length),
        elitism_count=2
    )
    
    # 3. Measure time and execute
    start_time = time.time()
    result = ga.run(population, problem.fitness_function, max_generations=max_gen)
    runtime = time.time() - start_time
    
    # 4. Print results to console
    print(f"Best Fitness: {result['best_fitness']}")
    print(f"Runtime: {runtime:.4f} seconds\n")
    
    # 5. Plot and save the convergence curve
    plt.figure(figsize=(10, 5))
    plt.plot(result["history"], label='Best Fitness', color='blue')
    plt.title(f'GA Convergence (OOP) - {problem_name}')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    
    # Dynamic file naming to prevent overwriting
    if "OneMax" in problem_name:
        base_name = "onemax"
    elif "Knapsack" in problem_name:
        base_name = "knapsack"
    elif "FeatureSelection" in problem_name:
        base_name = "feature_selection"
    else:
        base_name = "other_problem"

    filename = f"{base_name}_curve_oop.png"
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()
    
    return {
        "problem": problem_name,
        "best_fitness": result['best_fitness'],
        "runtime_seconds": runtime,
        "best_solution": result['best_chromosome'].genes,
        "history": result["history"]
    }

def main():
    results = {}
    
    # 1. Run OneMax
    onemax = OneMaxProblem(length=100)
    results["OneMax"] = run_experiment("OneMax", onemax, length=100)
    
    # 2. Run Knapsack
    knapsack = KnapsackProblem(num_items=100, seed=42)
    results["Knapsack"] = run_experiment("0/1 Knapsack", knapsack, length=100)
    
    # 3. [BONUS EXTENSION] Run Feature Selection problem
    feature_selection = FeatureSelectionProblem(total_features=100, seed=42)
    results["FeatureSelection"] = run_experiment("FeatureSelection", feature_selection, length=100)

    # Save all data to a JSON file
    json_path = os.path.join(REPORT_DIR, "results_oop.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Successfully saved plots and {json_path} to the reports/ directory")

if __name__ == "__main__":
    main()