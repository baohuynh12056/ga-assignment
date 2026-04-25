import unittest
import random
from src.problems import onemax_fitness, generate_random_genes
from src.operators import evaluate_population, tournament_selection, one_point_crossover, bitflip_mutation
from src.ga import run_ga

class TestFunctionalGA(unittest.TestCase):

    def setUp(self):
        # Fix the seed so that randomized tests always produce consistent results
        random.seed(42)

    def test_fitness_evaluation(self):
        """Test fitness calculation and the population map function."""
        # 1. Test the pure fitness function
        genes_5 = (1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
        genes_10 = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        self.assertEqual(onemax_fitness(genes_5), 5.0)
        self.assertEqual(onemax_fitness(genes_10), 10.0)

        # 2. Test mapping the entire population
        pop_genes = ((1, 0), (1, 1), (0, 0))
        evaluated = evaluate_population(pop_genes, onemax_fitness)
        
        # The result must be nested Tuples: ((genes, fitness), (genes, fitness), ...)
        expected = (((1, 0), 1.0), ((1, 1), 2.0), ((0, 0), 0.0))
        self.assertEqual(evaluated, expected)

    def test_selection(self):
        """Test selection: Must return a Tuple containing the selected genes (excluding fitness)."""
        evaluated_pop = (((1, 0), 1.0), ((1, 1), 2.0), ((0, 0), 0.0))
        
        # Run selection with k=2
        parents = tournament_selection(evaluated_pop, k=2)
        
        # The size of parents must equal the initial population size
        self.assertEqual(len(parents), 3)
        # The inner elements must be gene Tuples, not nested Tuples
        self.assertIsInstance(parents[0], tuple)
        self.assertIsInstance(parents[0][0], int)

    def test_crossover(self):
        """Test one-point crossover with 100% probability."""
        p1 = (1, 1, 1, 1)
        p2 = (0, 0, 0, 0)
        
        child1, child2 = one_point_crossover(p1, p2, prob=1.0)
        
        self.assertEqual(len(child1), 4)
        self.assertEqual(len(child2), 4)
        
        # The resulting offspring must contain both 0 and 1 bits
        self.assertTrue(0 in child1 and 1 in child1)
        self.assertTrue(0 in child2 and 1 in child2)
        
        # Ensure the original data is not modified (Immutability)
        self.assertEqual(p1, (1, 1, 1, 1))
        self.assertEqual(p2, (0, 0, 0, 0))

    def test_mutation(self):
        """Test bit-flip mutation with 100% probability."""
        genes = (0, 0, 0, 0)
        mutated = bitflip_mutation(genes, prob=1.0)
        
        self.assertEqual(mutated, (1, 1, 1, 1))
        # The original data must remain unchanged
        self.assertEqual(genes, (0, 0, 0, 0))

    def test_improvement_over_generations(self):
        """Test the entire execution flow of the algorithm (Integration test)."""
        # Small population and generation count for a quick test
        length = 10
        pop_size = 10
        initial_pop_genes = tuple(generate_random_genes(length) for _ in range(pop_size))
        
        # Best fitness before running
        initial_evaluated = evaluate_population(initial_pop_genes, onemax_fitness)
        initial_best = max(initial_evaluated, key=lambda ind: ind[1])[1]
        
        # Run GA
        result = run_ga(
            initial_pop_genes=initial_pop_genes,
            fitness_func=onemax_fitness,
            crossover_prob=0.9,
            mutation_prob=0.1,
            elitism_count=1,
            max_generations=15
        )
        
        final_best = result["best_fitness"]
        
        # The algorithm must be able to evolve (fitness must not decrease)
        self.assertGreaterEqual(final_best, initial_best)

if __name__ == '__main__':
    unittest.main()