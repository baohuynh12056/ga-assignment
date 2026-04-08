import random
from typing import Callable, List, Dict, Any
from .chromosome import Chromosome, Population
from .strategies import SelectionStrategy, CrossoverStrategy, MutationStrategy

class GeneticAlgorithm:
    def __init__(self, 
                 selection: SelectionStrategy, 
                 crossover: CrossoverStrategy, 
                 mutation: MutationStrategy,
                 elitism_count: int = 2):
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.elitism_count = elitism_count # Số lượng tinh hoa giữ lại 

    def run(self, initial_population: Population, fitness_function: Callable[[List[int]], float], max_generations: int) -> Dict[str, Any]:
        current_population = initial_population
        history_best_fitness = []

        for generation in range(max_generations):
            # 1. Đánh giá (Evaluate Fitness)
            for chromo in current_population.individuals:
                if chromo.fitness is None:
                    chromo.fitness = fitness_function(chromo.genes)

            # Lưu lại dữ liệu để vẽ biểu đồ
            best_chromo = current_population.get_best_individual()
            history_best_fitness.append(best_chromo.fitness)

            # 2. Chủ nghĩa tinh hoa (Elitism)
            # Sắp xếp giảm dần theo điểm fitness
            sorted_individuals = sorted(current_population.individuals, key=lambda c: c.fitness, reverse=True)
            elites = [c.clone() for c in sorted_individuals[:self.elitism_count]]

            # 3. Chọn lọc cha mẹ (Selection)
            parents = self.selection.select(current_population)

            # 4 & 5. Lai ghép và Đột biến (Crossover & Mutation)
            offspring = []
            # Duyệt qua từng cặp cha mẹ
            for i in range(0, len(parents) - 1, 2):
                if len(offspring) >= (current_population.size() - self.elitism_count):
                    break # Dừng lại nếu đã tạo đủ con cháu

                parent1 = parents[i]
                parent2 = parents[i+1]

                child1, child2 = self.crossover.crossover(parent1, parent2)
                
                self.mutation.mutate(child1)
                self.mutation.mutate(child2)

                offspring.extend([child1, child2])

            # Nếu danh sách con cháu bị dư 1 cá thể do lai ghép chẵn, ta sẽ cắt bớt
            offspring = offspring[:current_population.size() - self.elitism_count]

            # 6. Thay thế quần thể (Replacement)
            current_population = Population(elites + offspring)

        # Đánh giá thế hệ cuối cùng
        for chromo in current_population.individuals:
            if chromo.fitness is None:
                chromo.fitness = fitness_function(chromo.genes)
                
        final_best = current_population.get_best_individual()
        history_best_fitness.append(final_best.fitness)

        return {
            "best_chromosome": final_best,
            "best_fitness": final_best.fitness,
            "history": history_best_fitness
        }