import time
import json
import random
import os
import matplotlib.pyplot as plt

from src.chromosome import Chromosome, Population
from src.operators import TournamentSelection, OnePointCrossover, BitflipMutation
from src.problems import OneMaxProblem, KnapsackProblem, FeatureSelectionProblem
from src.ga import GeneticAlgorithm

# Đảm bảo thư mục reports tồn tại ở thư mục gốc chứa project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "reports"))
os.makedirs(REPORT_DIR, exist_ok=True)

def run_experiment(problem_name, problem, pop_size=100, max_gen=300, length=100):
    print(f"--- Đang giải bài toán {problem_name} ---")
    
    # Thiết lập seed cố định theo yêu cầu bài tập
    random.seed(42)
    
    # 1. Khởi tạo quần thể ban đầu
    initial_chromosomes = [Chromosome(problem.generate_random_genes()) for _ in range(pop_size)]
    population = Population(initial_chromosomes)
    
    # 2. Cấu hình GA
    ga = GeneticAlgorithm(
        selection=TournamentSelection(k=3),
        crossover=OnePointCrossover(probability=0.9),
        mutation=BitflipMutation(probability=1.0 / length),
        elitism_count=2
    )
    
    # 3. Đo thời gian và chạy
    start_time = time.time()
    result = ga.run(population, problem.fitness_function, max_generations=max_gen)
    runtime = time.time() - start_time
    
    # 4. In kết quả ra màn hình
    print(f"Điểm Fitness tốt nhất: {result['best_fitness']}")
    print(f"Thời gian chạy: {runtime:.4f} giây\n")
    
    # 5. Vẽ và lưu đồ thị (Curve)
    plt.figure(figsize=(10, 5))
    plt.plot(result["history"], label='Best Fitness', color='blue')
    plt.title(f'GA Convergence - {problem_name}')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    
    if "OneMax" in problem_name:
        base_name = "onemax"
    elif "Knapsack" in problem_name:
        base_name = "knapsack"
    elif "FeatureSelection" in problem_name:
        base_name = "feature_selection"
    else:
        base_name = "{other_problem}"

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
    
    # Chạy OneMax
    onemax = OneMaxProblem(length=100)
    results["OneMax"] = run_experiment("OneMax", onemax, length=100)
    
    # Chạy Knapsack
    knapsack = KnapsackProblem(num_items=100, seed=42)
    results["Knapsack"] = run_experiment("0/1 Knapsack", knapsack, length=100)
    
    # 3. [BONUS EXTENSION] Chạy bài toán Feature Selection
    feature_selection = FeatureSelectionProblem(total_features=100, seed=42)
    results["FeatureSelection"] = run_experiment("FeatureSelection", feature_selection, length=100)

    # Lưu toàn bộ dữ liệu ra file JSON
    json_path = os.path.join(REPORT_DIR, "results_oop.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Đã lưu thành công biểu đồ và file {json_path} vào thư mục reports/")

if __name__ == "__main__":
    main()