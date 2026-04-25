import random

# OneMax Problem
def onemax_fitness(genes: tuple) -> float:
    # Use reduce or sum to calculate the total
    return float(sum(genes))

def generate_random_genes(length: int) -> tuple:
    return tuple(random.choice((0, 1)) for _ in range(length))

# Knapsack Problem using Closure
def make_knapsack_fitness(num_items: int = 100, seed: int = 42):
    # Initialize fixed data inside the closure
    rng = random.Random(seed)
    weights = tuple(rng.randint(1, 20) for _ in range(num_items))
    values = tuple(rng.randint(10, 100) for _ in range(num_items))
    capacity = sum(weights) * 0.4
    
    # This is a pure function that will be returned for the GA to use
    def fitness_function(genes: tuple) -> float:
        total_weight = sum(w * g for w, g in zip(weights, genes))
        total_value = sum(v * g for v, g in zip(values, genes))
        
        if total_weight > capacity:
            return 0.0
        return float(total_value)
        
    return fitness_function

def make_feature_selection_fitness(total_features: int = 100, seed: int = 42):
    """
    Applying GA for Feature Selection in Machine Learning.
    Simulation: Given 100 features. 
    Objective: Maximize the total importance of selected features, 
    but apply a penalty for selecting too many features (encourages sparse models).
    """
    rng = random.Random(seed)
    
    # Create an importance list: 20 good features (5-10) and 80 noise features (-2 to 1)
    importances_list = [rng.uniform(5.0, 10.0) for _ in range(20)] + \
                       [rng.uniform(-2.0, 1.0) for _ in range(total_features - 20)]
    
    rng.shuffle(importances_list)
    
    # Cast to Tuple to ensure Immutability (Data cannot be modified)
    feature_importances = tuple(importances_list)
    
    # In practice, this can be replaced by using RandomForest to measure real feature importance:
    # rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
    # rf_model.fit(X_train, y_train)
    # importances_array = rf_model.feature_importances_ * 100
    # feature_importances = tuple(importances_array)

    # Pure function to be passed into the GA
    def fitness_function(genes: tuple) -> float:
        information_gain = sum(imp * g for imp, g in zip(feature_importances, genes))
        
        num_selected = sum(genes)
        complexity_penalty = num_selected * 1.5 # L0 Regularization Penalty
        
        final_score = information_gain - complexity_penalty
        return float(max(0.0, final_score))
        
    return fitness_function