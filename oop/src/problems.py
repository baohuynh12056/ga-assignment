import random
from typing import List

class OneMaxProblem:
    def __init__(self, length: int = 100):
        self.length = length # Length of the chromosome

    def fitness_function(self, genes: List[int]) -> float:
        return float(sum(genes)) # Score = number of 1 bits

    def generate_random_genes(self) -> List[int]:
        return [random.choice([0, 1]) for _ in range(self.length)]


class KnapsackProblem:
    def __init__(self, num_items: int = 100, seed: int = 42):
        # Randomly initialize weights and values using the same seed for reproducibility
        random.seed(seed)
        self.num_items = num_items 
        self.weights = [random.randint(1, 20) for _ in range(num_items)]
        self.values = [random.randint(10, 100) for _ in range(num_items)]
        self.capacity = sum(self.weights) * 0.4 # 40% capacity 
        random.seed() # Reset seed to avoid affecting the GA algorithm

    def fitness_function(self, genes: List[int]) -> float:
        total_weight = sum(w * g for w, g in zip(self.weights, genes))
        total_value = sum(v * g for v, g in zip(self.values, genes))
        
        if total_weight > self.capacity:
            return 0.0 # Penalized to 0 if capacity is exceeded 
        return float(total_value)

    def generate_random_genes(self) -> List[int]:
        return [random.choice([0, 1]) for _ in range(self.num_items)]
    
class FeatureSelectionProblem:
    """
    Applying GA for Feature Selection in Machine Learning.
    Simulation: Given 100 features. 
    Objective: Maximize the total importance of selected features, 
    but apply a penalty for selecting too many features (encourages sparse models).
    """
    def __init__(self, total_features: int = 100, seed: int = 42):
        self.total_features = total_features
        random.seed(seed)
        
        # Simulation: 20 good features (high scores from 5-10)
        # The remaining 80 features are noise (negative or very low scores from -2 to 1)
        self.feature_importances = [random.uniform(5.0, 10.0) for _ in range(20)] + \
                                   [random.uniform(-2.0, 1.0) for _ in range(total_features - 20)]
        
               
        # In practice, this can be replaced by using RandomForest to measure real feature importance:
        # rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        # rf_model.fit(X_train, y_train)
        # self.feature_importances = rf_model.feature_importances_
        # self.feature_importances = self.feature_importances * 100

        # Randomly shuffle the positions of the features
        random.shuffle(self.feature_importances)
        random.seed() # Reset seed

    def fitness_function(self, genes: List[int]) -> float:
        # Calculate total information gain of selected features (bit = 1)
        information_gain = sum(importance * gene for importance, gene in zip(self.feature_importances, genes))
        
        # L0 Regularization Penalty: Penalize 1.5 points for EACH selected feature
        # Forces the algorithm to weigh: "Does this feature provide > 1.5 points of benefit? If not, discard it (set = 0)"
        num_selected = sum(genes)
        complexity_penalty = num_selected * 1.5 
        
        final_score = information_gain - complexity_penalty
        
        # Ensure fitness score is not negative (GA usually performs better with positive fitness)
        return float(max(0.0, final_score))

    def generate_random_genes(self) -> List[int]:
        return [random.choice([0, 1]) for _ in range(self.total_features)]