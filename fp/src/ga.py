from functools import reduce
from typing import Callable
from .operators import evaluate_population, tournament_selection, one_point_crossover, bitflip_mutation

def run_ga(initial_pop_genes: tuple, 
           fitness_func: Callable, 
           crossover_prob: float, 
           mutation_prob: float, 
           elitism_count: int, 
           max_generations: int) -> dict:
    
    # 1. Đánh giá quần thể khởi tạo
    initial_evaluated = evaluate_population(initial_pop_genes, fitness_func)

    # 2. Định nghĩa hàm bước tiến hóa (Chạy cho mỗi thế hệ)
    def generation_step(state: dict, gen_index: int) -> dict:
        current_pop = state["population"]
        history = state["history"]

        # Lấy người giỏi nhất hiện tại
        best_ind = max(current_pop, key=lambda ind: ind[1])
        new_history = history + (best_ind[1],) # Nối tuple
        
        # Elitism
        sorted_pop = sorted(current_pop, key=lambda ind: ind[1], reverse=True)
        elites = tuple(ind[0] for ind in sorted_pop[:elitism_count])

        # Selection
        parents = tournament_selection(current_pop, k=3)

        # Crossover & Mutation bằng reduce để tích lũy dần thế hệ con cháu
        def produce_offspring(acc: tuple, i: int) -> tuple:
            if len(acc) >= len(current_pop) - elitism_count:
                return acc
            c1, c2 = one_point_crossover(parents[i], parents[i+1], crossover_prob)
            m1 = bitflip_mutation(c1, mutation_prob)
            m2 = bitflip_mutation(c2, mutation_prob)
            return acc + (m1, m2)

        indices = tuple(range(0, len(parents) - 1, 2))
        offspring = reduce(produce_offspring, indices, ())
        offspring = offspring[:len(current_pop) - elitism_count] # Cắt phần thừa

        # Ghép Tinh hoa và Con cháu để ra quần thể gen thế hệ mới
        next_genes_pop = elites + offspring
        next_evaluated_pop = evaluate_population(next_genes_pop, fitness_func)

        return {"population": next_evaluated_pop, "history": new_history}

    # 3. Chạy vòng lặp tiến hóa bằng reduce (Thay cho vòng lặp for)
    initial_state = {"population": initial_evaluated, "history": ()}
    final_state = reduce(generation_step, range(max_generations), initial_state)

    # 4. Trích xuất kết quả cuối cùng
    final_pop = final_state["population"]
    final_best = max(final_pop, key=lambda ind: ind[1])
    final_history = final_state["history"] + (final_best[1],)

    return {
        "best_genes": final_best[0],
        "best_fitness": final_best[1],
        "history": list(final_history) # Chuyển về list để dump JSON 
    }