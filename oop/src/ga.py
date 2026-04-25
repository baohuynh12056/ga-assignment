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
        self.elitism_count = elitism_count # Number of elite individuals to keep

    def run(self, initial_population: Population, fitness_function: Callable[[List[int]], float], max_generations: int) -> Dict[str, Any]:
        current_population = initial_population
        history_best_fitness = []

        for generation in range(max_generations):
            # 1. Evaluate Fitness
            for chromo in current_population.individuals:
                if chromo.fitness is None:
                    chromo.fitness = fitness_function(chromo.genes)

            # Save data for plotting
            best_chromo = current_population.get_best_individual()
            history_best_fitness.append(best_chromo.fitness)

            # 2. Elitism
            # Sort in descending order based on fitness score
            sorted_individuals = sorted(current_population.individuals, key=lambda c: c.fitness, reverse=True)
            elites = [c.clone() for c in sorted_individuals[:self.elitism_count]]

            # 3. Parent Selection
            parents = self.selection.select(current_population)

            # 4 & 5. Crossover & Mutation
            offspring = []
            # Iterate through parent pairs
            for i in range(0, len(parents) - 1, 2):
                if len(offspring) >= (current_population.size() - self.elitism_count):
                    break # Stop if enough offspring have been created

                parent1 = parents[i]
                parent2 = parents[i+1]

                child1, child2 = self.crossover.crossover(parent1, parent2)
                
                self.mutation.mutate(child1)
                self.mutation.mutate(child2)

                offspring.extend([child1, child2])

            # Trim offspring list if it exceeds the required size due to paired crossover
            offspring = offspring[:current_population.size() - self.elitism_count]

            # 6. Population Replacement
            current_population = Population(elites + offspring)

        # Evaluate the final generation
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