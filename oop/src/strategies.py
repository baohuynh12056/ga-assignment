from abc import ABC, abstractmethod
from typing import List, Tuple

class SelectionStrategy(ABC):
    """Interface cho chiến lược chọn lọc cha mẹ."""
    @abstractmethod
    def select(self, population: 'Population') -> List['Chromosome']:
        pass

class CrossoverStrategy(ABC):
    """Interface cho chiến lược lai ghép."""
    @abstractmethod
    def crossover(self, parent1: 'Chromosome', parent2: 'Chromosome') -> Tuple['Chromosome', 'Chromosome']:
        pass

class MutationStrategy(ABC):
    """Interface cho chiến lược đột biến."""
    @abstractmethod
    def mutate(self, chromosome: 'Chromosome') -> None:
        pass