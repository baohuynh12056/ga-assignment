import random

# Bài toán One Max
def onemax_fitness(genes: tuple) -> float:
    # Dùng reduce hoặc sum để tính tổng
    return float(sum(genes))

def generate_random_genes(length: int) -> tuple:
    return tuple(random.choice((0, 1)) for _ in range(length))

# Bài toán Knapsack sử dụng Closure (Hàm bao đóng)
def make_knapsack_fitness(num_items: int = 100, seed: int = 42):
    # Khởi tạo data cố định bên trong hàm bao
    rng = random.Random(seed)
    weights = tuple(rng.randint(1, 20) for _ in range(num_items))
    values = tuple(rng.randint(10, 100) for _ in range(num_items))
    capacity = sum(weights) * 0.4
    
    # Đây là hàm thuần túy (pure function) sẽ được trả về để thuật toán GA sử dụng
    def fitness_function(genes: tuple) -> float:
        total_weight = sum(w * g for w, g in zip(weights, genes))
        total_value = sum(v * g for v, g in zip(values, genes))
        
        if total_weight > capacity:
            return 0.0
        return float(total_value)
        
    return fitness_function

def make_feature_selection_fitness(total_features: int = 100, seed: int = 42):
    """
    Ứng dụng GA để chọn lọc đặc trưng (Feature Selection) trong Machine Learning.
    Giả lập: Có 100 features. 
    Mục tiêu: Tối đa hóa tổng độ quan trọng của các feature được chọn, 
    nhưng bị phạt (penalty) nếu chọn quá nhiều feature (giúp mô hình tinh gọn).
    """
    rng = random.Random(seed)
    
    # Tạo danh sách điểm: 20 features tốt (5-10) và 80 features nhiễu (-2 đến 1)
    importances_list = [rng.uniform(5.0, 10.0) for _ in range(20)] + \
                       [rng.uniform(-2.0, 1.0) for _ in range(total_features - 20)]
    
    rng.shuffle(importances_list)
    
    # Ép kiểu sang Tuple để đảm bảo Immutability (Không ai có thể sửa data này)
    feature_importances = tuple(importances_list)
    
    # Có thể thay thế bằng cho xác với thực tế dùng RandomForest đo lường độ quan trọng của feature
    # rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
    # rf_model.fit(X_train, y_train)
    # importances_array = rf_model.feature_importances_ * 100
    # feature_importances = tuple(importances_array)

    # Hàm thuần túy (Pure function) sẽ được truyền vào GA
    def fitness_function(genes: tuple) -> float:
        information_gain = sum(imp * g for imp, g in zip(feature_importances, genes))
        
        num_selected = sum(genes)
        complexity_penalty = num_selected * 1.5 # Phạt L0 Regularization
        
        final_score = information_gain - complexity_penalty
        return float(max(0.0, final_score))
        
    return fitness_function