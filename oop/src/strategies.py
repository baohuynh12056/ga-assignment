from abc import ABC, abstractmethod
from typing import List, Tuple

class SelectionStrategy(ABC):
    """Interface for parent selection strategy."""
    @abstractmethod
    def select(self, population: 'Population') -> List['Chromosome']:
        pass

class CrossoverStrategy(ABC):
    """Interface for crossover strategy."""
    @abstractmethod
    def crossover(self, parent1: 'Chromosome', parent2: 'Chromosome') -> Tuple['Chromosome', 'Chromosome']:
        pass

class MutationStrategy(ABC):
    """Interface for mutation strategy."""
    @abstractmethod
    def mutate(self, chromosome: 'Chromosome') -> None:
        pass