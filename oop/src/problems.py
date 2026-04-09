import random
from typing import List

class OneMaxProblem:
    def __init__(self, length: int = 100):
        self.length = length # 

    def fitness_function(self, genes: List[int]) -> float:
        return float(sum(genes)) # Điểm = số lượng bit 1

    def generate_random_genes(self) -> List[int]:
        return [random.choice([0, 1]) for _ in range(self.length)]


class KnapsackProblem:
    def __init__(self, num_items: int = 100, seed: int = 42):
        # Khởi tạo ngẫu nhiên weights và values bằng chung 1 seed để dễ test 
        random.seed(seed)
        self.num_items = num_items # 
        self.weights = [random.randint(1, 20) for _ in range(num_items)]
        self.values = [random.randint(10, 100) for _ in range(num_items)]
        self.capacity = sum(self.weights) * 0.4 # Sức chứa 40% 
        random.seed() # Reset seed để không ảnh hưởng đến thuật toán GA

    def fitness_function(self, genes: List[int]) -> float:
        total_weight = sum(w * g for w, g in zip(self.weights, genes))
        total_value = sum(v * g for v, g in zip(self.values, genes))
        
        if total_weight > self.capacity:
            return 0.0 # Bị phạt về 0 nếu vượt quá sức chứa 
        return float(total_value)

    def generate_random_genes(self) -> List[int]:
        return [random.choice([0, 1]) for _ in range(self.num_items)]