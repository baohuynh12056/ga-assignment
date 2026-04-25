import random
from typing import Callable, Tuple

def evaluate_population(genes_pop: tuple, fitness_func: Callable) -> tuple:
    """Transforms a Tuple of genes into a Tuple containing (genes, fitness_score) using map."""
    return tuple(map(lambda genes: (genes, fitness_func(genes)), genes_pop))

def tournament_selection(evaluated_pop: tuple, k: int = 3) -> tuple:
    """Selects parents. Returns a Tuple containing the selected gene sequences."""
    def select_one(_) -> tuple:
        tournament = random.sample(evaluated_pop, k)
        # max() automatically finds the tuple with the highest fitness score (index 1), then extracts the genes (index 0)
        return max(tournament, key=lambda ind: ind[1])[0]
    
    # Use map to repeat the selection process matching the population size
    return tuple(map(select_one, range(len(evaluated_pop))))

def one_point_crossover(parent1: tuple, parent2: tuple, prob: float = 0.9) -> Tuple[tuple, tuple]:
    """Performs one-point crossover, returning 2 new gene tuples."""
    if random.random() < prob:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

def bitflip_mutation(genes: tuple, prob: float) -> tuple:
    """Performs bit-flip mutation using map."""
    return tuple(map(lambda g: 1 - g if random.random() < prob else g, genes))