import random

# Bài toán One Max
def onemax_fitness(genes: tuple) -> float:
    # Dùng reduce hoặc sum để tính tổng
    return float(sum(genes))

def generate_random_genes(length: int) -> tuple:
    return tuple(random.choice((0, 1)) for _ in range(length))

# Bài toán Knapsack sử dụng Closure (Hàm bao đóng)
def make_knapsack_fitness(num_items: int = 100, seed: int = 42):
    # Khởi tạo data cố định bên trong hàm bao
    rng = random.Random(seed)
    weights = tuple(rng.randint(1, 20) for _ in range(num_items))
    values = tuple(rng.randint(10, 100) for _ in range(num_items))
    capacity = sum(weights) * 0.4
    
    # Đây là hàm thuần túy (pure function) sẽ được trả về để thuật toán GA sử dụng
    def fitness_function(genes: tuple) -> float:
        total_weight = sum(w * g for w, g in zip(weights, genes))
        total_value = sum(v * g for v, g in zip(values, genes))
        
        if total_weight > capacity:
            return 0.0
        return float(total_value)
        
    return fitness_function