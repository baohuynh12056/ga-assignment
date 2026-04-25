import random
from typing import List, Tuple
from .strategies import SelectionStrategy, CrossoverStrategy, MutationStrategy
from .chromosome import Chromosome, Population

class TournamentSelection(SelectionStrategy):
    def __init__(self, k: int = 3):
        self.k = k # Tournament size 

    def select(self, population: Population) -> List[Chromosome]:
        parents = []
        individuals = population.individuals
        # Select enough parents to match the population size
        for _ in range(population.size()):
            tournament = random.sample(individuals, self.k)
            best = max(tournament, key=lambda c: c.fitness)
            parents.append(best)
        return parents


class OnePointCrossover(CrossoverStrategy):
    def __init__(self, probability: float = 0.9):
        self.probability = probability # Crossover probability 

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        if random.random() < self.probability:
            # Choose a random crossover point from index 1 to len-1
            point = random.randint(1, len(parent1) - 1)
            genes1 = parent1.genes[:point] + parent2.genes[point:]
            genes2 = parent2.genes[:point] + parent1.genes[point:]
            return Chromosome(genes1), Chromosome(genes2)
        
        # If no crossover occurs, return clones of the parents
        return parent1.clone(), parent2.clone()


class BitflipMutation(MutationStrategy):
    def __init__(self, probability: float = None):
        # If not provided, it defaults to 1/L 
        self.probability = probability 

    def mutate(self, chromosome: Chromosome) -> None:
        prob = self.probability if self.probability is not None else 1.0 / len(chromosome)
        new_genes = []
        for gene in chromosome.genes:
            if random.random() < prob:
                new_genes.append(1 - gene) # Flip bit: 0->1 or 1->0
            else:
                new_genes.append(gene)
        chromosome.genes = new_genes