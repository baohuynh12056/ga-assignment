import random
from typing import Callable, Tuple

def evaluate_population(genes_pop: tuple, fitness_func: Callable) -> tuple:
    """Biến Tuple các gen thành Tuple chứa (gen, điểm_fitness) bằng hàm map."""
    return tuple(map(lambda genes: (genes, fitness_func(genes)), genes_pop))

def tournament_selection(evaluated_pop: tuple, k: int = 3) -> tuple:
    """Chọn lọc cha mẹ. Trả về Tuple chứa các bộ gen được chọn."""
    def select_one(_) -> tuple:
        tournament = random.sample(evaluated_pop, k)
        # max() tự động lấy tuple có điểm fitness (index 1) cao nhất, sau đó lấy ra bộ gen (index 0)
        return max(tournament, key=lambda ind: ind[1])[0]
    
    # Dùng map để lặp việc chọn lọc bằng với kích thước quần thể
    return tuple(map(select_one, range(len(evaluated_pop))))

def one_point_crossover(parent1: tuple, parent2: tuple, prob: float = 0.9) -> Tuple[tuple, tuple]:
    """Lai ghép 1 điểm, trả về 2 tuple gen mới."""
    if random.random() < prob:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

def bitflip_mutation(genes: tuple, prob: float) -> tuple:
    """Lật bit bằng map."""
    return tuple(map(lambda g: 1 - g if random.random() < prob else g, genes))