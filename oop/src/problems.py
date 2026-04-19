import random
from typing import List

class OneMaxProblem:
    def __init__(self, length: int = 100):
        self.length = length # 

    def fitness_function(self, genes: List[int]) -> float:
        return float(sum(genes)) # Điểm = số lượng bit 1

    def generate_random_genes(self) -> List[int]:
        return [random.choice([0, 1]) for _ in range(self.length)]


class KnapsackProblem:
    def __init__(self, num_items: int = 100, seed: int = 42):
        # Khởi tạo ngẫu nhiên weights và values bằng chung 1 seed để dễ test 
        random.seed(seed)
        self.num_items = num_items # 
        self.weights = [random.randint(1, 20) for _ in range(num_items)]
        self.values = [random.randint(10, 100) for _ in range(num_items)]
        self.capacity = sum(self.weights) * 0.4 # Sức chứa 40% 
        random.seed() # Reset seed để không ảnh hưởng đến thuật toán GA

    def fitness_function(self, genes: List[int]) -> float:
        total_weight = sum(w * g for w, g in zip(self.weights, genes))
        total_value = sum(v * g for v, g in zip(self.values, genes))
        
        if total_weight > self.capacity:
            return 0.0 # Bị phạt về 0 nếu vượt quá sức chứa 
        return float(total_value)

    def generate_random_genes(self) -> List[int]:
        return [random.choice([0, 1]) for _ in range(self.num_items)]
    
class FeatureSelectionProblem:
    """
    Ứng dụng GA để chọn lọc đặc trưng (Feature Selection) trong Machine Learning.
    Giả lập: Có 100 features. 
    Mục tiêu: Tối đa hóa tổng độ quan trọng của các feature được chọn, 
    nhưng bị phạt (penalty) nếu chọn quá nhiều feature (giúp mô hình tinh gọn).
    """
    def __init__(self, total_features: int = 100, seed: int = 42):
        self.total_features = total_features
        random.seed(seed)
        
        # Giả lập: 20 features mang tín hiệu tốt (điểm cao từ 5-10)
        # 80 features còn lại là nhiễu (điểm âm hoặc rất thấp từ -2 đến 1)
        self.feature_importances = [random.uniform(5.0, 10.0) for _ in range(20)] + \
                                   [random.uniform(-2.0, 1.0) for _ in range(total_features - 20)]
        
               
        # Có thể thay thế bằng cho xác với thực tế dùng RandomForest đo lường độ quan trọng của feature
        # rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        # rf_model.fit(X_train, y_train)
        # self.feature_importances = rf_model.feature_importances_
        # self.feature_importances = self.feature_importances * 100

        # Xáo trộn ngẫu nhiên vị trí các features
        random.shuffle(self.feature_importances)
        random.seed() # Reset seed

    def fitness_function(self, genes: List[int]) -> float:
        # Tính tổng giá trị thông tin của các features được chọn (bit = 1)
        information_gain = sum(importance * gene for importance, gene in zip(self.feature_importances, genes))
        
        # L0 Regularization Penalty: Phạt 1.5 điểm cho MỖI feature được chọn
        # Ép thuật toán phải cân nhắc: "Feature này có mang lại lợi ích > 1.5 điểm không? Nếu không thì bỏ đi (set = 0)"
        num_selected = sum(genes)
        complexity_penalty = num_selected * 1.5 
        
        final_score = information_gain - complexity_penalty
        
        # Đảm bảo điểm fitness không bị âm (GA thường hoạt động tốt hơn với fitness dương)
        return float(max(0.0, final_score))

    def generate_random_genes(self) -> List[int]:
        return [random.choice([0, 1]) for _ in range(self.total_features)]