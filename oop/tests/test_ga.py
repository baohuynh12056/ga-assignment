import unittest
import random
from src.chromosome import Chromosome, Population
from src.operators import TournamentSelection, OnePointCrossover, BitflipMutation
from src.problems import OneMaxProblem
from src.ga import GeneticAlgorithm

class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        # Setup to run before each test to ensure consistency
        random.seed(42)

    def test_fitness_evaluation(self):
        """Test if the OneMax fitness function correctly counts the number of 1 bits."""
        problem = OneMaxProblem(length=10)
        genes_5 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        genes_10 = [1] * 10
        
        self.assertEqual(problem.fitness_function(genes_5), 5.0)
        self.assertEqual(problem.fitness_function(genes_10), 10.0)

    def test_selection(self):
        """Test Selection: Must return the exact number of parents equal to the population size."""
        pop = Population([Chromosome([1,0]), Chromosome([0,1]), Chromosome([1,1])])
        for c in pop.individuals:
            c.fitness = sum(c.genes) # Simulate fitness scores
            
        selection = TournamentSelection(k=2)
        parents = selection.select(pop)
        
        self.assertEqual(len(parents), 3) # For a population of 3, the number of selected parents must also be 3
        self.assertIsInstance(parents[0], Chromosome)

    def test_crossover(self):
        """Test Crossover: With 100% probability, offspring must maintain the same length."""
        crossover = OnePointCrossover(probability=1.0)
        p1 = Chromosome([1, 1, 1, 1])
        p2 = Chromosome([0, 0, 0, 0])
        
        child1, child2 = crossover.crossover(p1, p2)
        
        self.assertEqual(len(child1), 4)
        self.assertEqual(len(child2), 4)
        # Because of one-point crossover, the offspring will definitely have a mix of 1s and 0s
        self.assertTrue(0 in child1.genes and 1 in child1.genes)

    def test_mutation(self):
        """Test Mutation: With 100% probability, all bits must be flipped."""
        mutation = BitflipMutation(probability=1.0) # 100% mutation
        chromo = Chromosome([0, 0, 0, 0])
        mutation.mutate(chromo)
        
        self.assertEqual(chromo.genes, [1, 1, 1, 1]) # All 0s must become 1s

    def test_improvement_over_generations(self):
        """Test Evolution: The final generation's score must be greater than or equal to the initial generation."""
        problem = OneMaxProblem(length=20)
        chromosomes = [Chromosome(problem.generate_random_genes()) for _ in range(10)]
        initial_pop = Population(chromosomes)
        
        ga = GeneticAlgorithm(
            selection=TournamentSelection(k=3),
            crossover=OnePointCrossover(probability=0.9),
            mutation=BitflipMutation(probability=0.05),
            elitism_count=1
        )
        
        # Get the best score before running
        for c in initial_pop.individuals:
            c.fitness = problem.fitness_function(c.genes)
        initial_best = initial_pop.get_best_individual().fitness
        
        # Run for 10 generations
        result = ga.run(initial_pop, problem.fitness_function, max_generations=10)
        final_best = result["best_fitness"]
        
        self.assertGreaterEqual(final_best, initial_best)

if __name__ == '__main__':
    unittest.main()