import unittest
import random
from src.problems import onemax_fitness, generate_random_genes
from src.operators import evaluate_population, tournament_selection, one_point_crossover, bitflip_mutation
from src.ga import run_ga

class TestFunctionalGA(unittest.TestCase):

    def setUp(self):
        # Cố định seed để các phép test có yếu tố ngẫu nhiên luôn ra kết quả giống nhau
        random.seed(42)

    def test_fitness_evaluation(self):
        """Kiểm tra tính điểm và hàm map quần thể."""
        # 1. Test hàm tính điểm thuần túy
        genes_5 = (1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
        genes_10 = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        self.assertEqual(onemax_fitness(genes_5), 5.0)
        self.assertEqual(onemax_fitness(genes_10), 10.0)

        # 2. Test hàm map toàn bộ quần thể
        pop_genes = ((1, 0), (1, 1), (0, 0))
        evaluated = evaluate_population(pop_genes, onemax_fitness)
        
        # Kết quả phải là Tuple lồng nhau: ((gen, fitness), (gen, fitness), ...)
        expected = (((1, 0), 1.0), ((1, 1), 2.0), ((0, 0), 0.0))
        self.assertEqual(evaluated, expected)

    def test_selection(self):
        """Kiểm tra chọn lọc: Phải trả về Tuple chứa các gen được chọn (bỏ fitness đi)."""
        evaluated_pop = (((1, 0), 1.0), ((1, 1), 2.0), ((0, 0), 0.0))
        
        # Chạy chọn lọc với k=2
        parents = tournament_selection(evaluated_pop, k=2)
        
        # Kích thước cha mẹ phải bằng kích thước quần thể ban đầu
        self.assertEqual(len(parents), 3)
        # Các phần tử bên trong phải là Tuple gen, không còn là Tuple lồng
        self.assertIsInstance(parents[0], tuple)
        self.assertIsInstance(parents[0][0], int)

    def test_crossover(self):
        """Kiểm tra lai ghép 1 điểm với xác suất 100%."""
        p1 = (1, 1, 1, 1)
        p2 = (0, 0, 0, 0)
        
        child1, child2 = one_point_crossover(p1, p2, prob=1.0)
        
        self.assertEqual(len(child1), 4)
        self.assertEqual(len(child2), 4)
        
        # Con lai sinh ra phải có cả bit 0 và bit 1
        self.assertTrue(0 in child1 and 1 in child1)
        self.assertTrue(0 in child2 and 1 in child2)
        
        # Đảm bảo dữ liệu gốc không bị thay đổi (Immutability)
        self.assertEqual(p1, (1, 1, 1, 1))
        self.assertEqual(p2, (0, 0, 0, 0))

    def test_mutation(self):
        """Kiểm tra đột biến lật bit với xác suất 100%."""
        genes = (0, 0, 0, 0)
        mutated = bitflip_mutation(genes, prob=1.0)
        
        self.assertEqual(mutated, (1, 1, 1, 1))
        # Dữ liệu gốc vẫn phải giữ nguyên
        self.assertEqual(genes, (0, 0, 0, 0))

    def test_improvement_over_generations(self):
        """Kiểm tra toàn bộ luồng chạy của thuật toán (Integration test)."""
        # Quần thể nhỏ, số thế hệ nhỏ để test nhanh
        length = 10
        pop_size = 10
        initial_pop_genes = tuple(generate_random_genes(length) for _ in range(pop_size))
        
        # Điểm tốt nhất trước khi chạy
        initial_evaluated = evaluate_population(initial_pop_genes, onemax_fitness)
        initial_best = max(initial_evaluated, key=lambda ind: ind[1])[1]
        
        # Chạy GA
        result = run_ga(
            initial_pop_genes=initial_pop_genes,
            fitness_func=onemax_fitness,
            crossover_prob=0.9,
            mutation_prob=0.1,
            elitism_count=1,
            max_generations=15
        )
        
        final_best = result["best_fitness"]
        
        # Thuật toán phải có khả năng tiến hóa (điểm không được giảm)
        self.assertGreaterEqual(final_best, initial_best)

if __name__ == '__main__':
    unittest.main()