import random
from typing import List, Tuple
from .strategies import SelectionStrategy, CrossoverStrategy, MutationStrategy
from .chromosome import Chromosome, Population

class TournamentSelection(SelectionStrategy):
    def __init__(self, k: int = 3):
        self.k = k # Kích thước giải đấu 

    def select(self, population: Population) -> List[Chromosome]:
        parents = []
        individuals = population.individuals
        # Chọn đủ số lượng cha mẹ bằng với kích thước quần thể
        for _ in range(population.size()):
            tournament = random.sample(individuals, self.k)
            best = max(tournament, key=lambda c: c.fitness)
            parents.append(best)
        return parents


class OnePointCrossover(CrossoverStrategy):
    def __init__(self, probability: float = 0.9):
        self.probability = probability # Xác suất lai ghép 

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        if random.random() < self.probability:
            # Chọn 1 điểm cắt ngẫu nhiên từ index 1 đến len-1
            point = random.randint(1, len(parent1) - 1)
            genes1 = parent1.genes[:point] + parent2.genes[point:]
            genes2 = parent2.genes[:point] + parent1.genes[point:]
            return Chromosome(genes1), Chromosome(genes2)
        
        # Nếu không lai ghép, trả về bản sao của cha mẹ
        return parent1.clone(), parent2.clone()


class BitflipMutation(MutationStrategy):
    def __init__(self, probability: float = None):
        # Nếu không truyền vào, sẽ tính mặc định là 1/L 
        self.probability = probability 

    def mutate(self, chromosome: Chromosome) -> None:
        prob = self.probability if self.probability is not None else 1.0 / len(chromosome)
        new_genes = []
        for gene in chromosome.genes:
            if random.random() < prob:
                new_genes.append(1 - gene) # Đảo bit: 0->1 hoặc 1->0
            else:
                new_genes.append(gene)
        chromosome.genes = new_genes