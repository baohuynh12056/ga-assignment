import unittest
import random
from src.chromosome import Chromosome, Population
from src.operators import TournamentSelection, OnePointCrossover, BitflipMutation
from src.problems import OneMaxProblem
from src.ga import GeneticAlgorithm

class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        # Thiết lập chạy trước mỗi bài test để đảm bảo tính nhất quán
        random.seed(42)

    def test_fitness_evaluation(self):
        """Kiểm tra xem hàm tính điểm OneMax có đếm đúng số bit 1 không."""
        problem = OneMaxProblem(length=10)
        genes_5 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        genes_10 = [1] * 10
        
        self.assertEqual(problem.fitness_function(genes_5), 5.0)
        self.assertEqual(problem.fitness_function(genes_10), 10.0)

    def test_selection(self):
        """Kiểm tra Chọn lọc: Phải trả về đúng số lượng cha mẹ bằng size quần thể."""
        pop = Population([Chromosome([1,0]), Chromosome([0,1]), Chromosome([1,1])])
        for c in pop.individuals:
            c.fitness = sum(c.genes) # Giả lập điểm fitness
            
        selection = TournamentSelection(k=2)
        parents = selection.select(pop)
        
        self.assertEqual(len(parents), 3) # Quần thể 3 thì số cha mẹ chọn ra cũng phải là 3
        self.assertIsInstance(parents[0], Chromosome)

    def test_crossover(self):
        """Kiểm tra Lai ghép: Nếu xác suất 100%, con sinh ra phải có chiều dài không đổi."""
        crossover = OnePointCrossover(probability=1.0)
        p1 = Chromosome([1, 1, 1, 1])
        p2 = Chromosome([0, 0, 0, 0])
        
        child1, child2 = crossover.crossover(p1, p2)
        
        self.assertEqual(len(child1), 4)
        self.assertEqual(len(child2), 4)
        # Vì lai 1 điểm, đứa con chắc chắn sẽ có sự pha trộn giữa 1 và 0
        self.assertTrue(0 in child1.genes and 1 in child1.genes)

    def test_mutation(self):
        """Kiểm tra Đột biến: Nếu xác suất 100%, tất cả bit phải bị lật."""
        mutation = BitflipMutation(probability=1.0) # 100% đột biến
        chromo = Chromosome([0, 0, 0, 0])
        mutation.mutate(chromo)
        
        self.assertEqual(chromo.genes, [1, 1, 1, 1]) # Tất cả 0 phải thành 1

    def test_improvement_over_generations(self):
        """Kiểm tra Tính tiến hóa: Điểm số thế hệ cuối phải lớn hơn hoặc bằng thế hệ đầu."""
        problem = OneMaxProblem(length=20)
        chromosomes = [Chromosome(problem.generate_random_genes()) for _ in range(10)]
        initial_pop = Population(chromosomes)
        
        ga = GeneticAlgorithm(
            selection=TournamentSelection(k=3),
            crossover=OnePointCrossover(probability=0.9),
            mutation=BitflipMutation(probability=0.05),
            elitism_count=1
        )
        
        # Lấy điểm tốt nhất trước khi chạy
        for c in initial_pop.individuals:
            c.fitness = problem.fitness_function(c.genes)
        initial_best = initial_pop.get_best_individual().fitness
        
        # Chạy 10 thế hệ
        result = ga.run(initial_pop, problem.fitness_function, max_generations=10)
        final_best = result["best_fitness"]
        
        self.assertGreaterEqual(final_best, initial_best)

if __name__ == '__main__':
    unittest.main()