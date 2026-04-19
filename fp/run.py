import time
import json
import random
import os
import matplotlib.pyplot as plt

from src.problems import onemax_fitness, make_knapsack_fitness, generate_random_genes, make_feature_selection_fitness
from src.ga import run_ga

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "reports"))
os.makedirs(REPORT_DIR, exist_ok=True)

def run_experiment(problem_name, fitness_func, length=100, pop_size=100, max_gen=300):
    print(f"--- Đang giải bài toán {problem_name} bằng FP ---")
    
    random.seed(42)
    
    # Khởi tạo dữ liệu gen đầu vào (Dùng Tuple để đảm bảo Immutability)
    initial_pop_genes = tuple(generate_random_genes(length) for _ in range(pop_size))
    
    start_time = time.time()
    
    # Gọi hàm xử lý luồng GA
    result = run_ga(
        initial_pop_genes=initial_pop_genes,
        fitness_func=fitness_func,
        crossover_prob=0.9,
        mutation_prob=1.0 / length,
        elitism_count=2,
        max_generations=max_gen
    )
    
    runtime = time.time() - start_time
    
    print(f"Điểm Fitness tốt nhất: {result['best_fitness']}")
    print(f"Thời gian chạy: {runtime:.4f} giây\n")
    
    # Vẽ đồ thị
    plt.figure(figsize=(10, 5))
    plt.plot(result["history"], label='Best Fitness', color='green')
    plt.title(f'GA Convergence (FP) - {problem_name}')
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

    filename = f"{base_name}_curve_fp.png"
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()
    
    return {
        "problem": problem_name,
        "best_fitness": result['best_fitness'],
        "runtime_seconds": runtime,
        "best_solution": result['best_genes'],
        "history": result["history"]
    }

def main():
    results = {}
    
    # OneMax
    results["OneMax"] = run_experiment("OneMax", onemax_fitness)
    
    # Knapsack
    knapsack_fitness = make_knapsack_fitness(num_items=100, seed=42)
    results["Knapsack"] = run_experiment("0/1 Knapsack", knapsack_fitness)
    
    # 3. [BONUS EXTENSION] Chạy bài toán Feature Selection (FP)
    feature_selection_fitness = make_feature_selection_fitness(total_features=100, seed=42)
    results["FeatureSelection"] = run_experiment("FeatureSelection", feature_selection_fitness, length=100)

    # Lưu file JSON
    json_path = os.path.join(REPORT_DIR, "results_fp.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Đã lưu kết quả FP vào {REPORT_DIR}")

if __name__ == "__main__":
    main()