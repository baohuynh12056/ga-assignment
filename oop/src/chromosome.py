from typing import List

class Chromosome:
    def __init__(self, genes: List[int]):
        self._genes = genes
        self._fitness = None

    @property
    def genes(self) -> List[int]:
        return self._genes

    @genes.setter
    def genes(self, new_genes: List[int]):
        self._genes = new_genes
        self._fitness = None # Reset fitness nếu gen bị thay đổi

    @property
    def fitness(self) -> float:
        return self._fitness

    @fitness.setter
    def fitness(self, value: float):
        self._fitness = value

    def clone(self) -> 'Chromosome':
        """Tạo ra một bản sao độc lập của nhiễm sắc thể."""
        clone_chromo = Chromosome(self._genes.copy())
        clone_chromo.fitness = self._fitness
        return clone_chromo

    def __len__(self):
        return len(self._genes)


class Population:
    def __init__(self, chromosomes: List[Chromosome]):
        self._individuals = chromosomes

    @property
    def individuals(self) -> List[Chromosome]:
        return self._individuals

    def get_best_individual(self) -> Chromosome:
        # Lọc ra những cá thể đã được đánh giá fitness
        evaluated = [c for c in self._individuals if c.fitness is not None]
        if not evaluated:
            raise ValueError("Chưa có cá thể nào được đánh giá điểm fitness.")
        return max(evaluated, key=lambda chromo: chromo.fitness)

    def size(self) -> int:
        return len(self._individuals)