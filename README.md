# **[Extended Assignment]** Genetic Algorithm (GA) — Object-Oriented vs Functional Programming

**Instructor:** Nguyen Thanh Cong, PhD

**Student Name:** Huynh Gia Bao

**Student ID:** 2410233

## 1. Project Overview

This project is an extended major assignment exploring the implementation of a Genetic Algorithm (GA) through two distinct software engineering paradigms: **Object-Oriented Programming (OOP)** and **Functional Programming (FP)**.

The primary objective is to evaluate the trade-offs between mutability and immutability, statefulness and pure functions, and execution speed versus code safety. To demonstrate the robustness and extensibility of the architecture, the GA is applied to three distinct optimization problems, ranging from classical computer science puzzles to practical machine learning applications.

## 2. Implemented Optimization Problems

1. **OneMax Problem:** The baseline test. A pure discrete optimization task aiming to maximize the number of `1`s in a binary chromosome of length `L=100`.
2. **0/1 Knapsack Problem:** A constrained combinatorial optimization problem. The GA must maximize the total value of items without exceeding a strict weight capacity (`n=100`).
3. **Feature Selection Simulation (Bonus - Extensible Design):** Applying GA to the Machine Learning domain. The algorithm selects an optimal subset of features to maximize total Information Gain based on a simulated importance landscape (20 highly informative features and 80 noise features generated via uniform random sampling), while incorporating an L0 regularization penalty to discourage selecting too many features and promote sparse, efficient models. In practical applications, this simulated importance can be replaced with real feature importance scores (e.g., from a Random Forest model).
## 3. Installation & Execution

### Prerequisites

- Python 3.8+
- `matplotlib` (for generating evolution curves)


```bash
pip install -r requirements.txt
```
### Running the Experiments
Both paradigms utilize identical hyperparameter constraints (Population: 100, Generations: 300, Mutation Rate: 1/L) controlled via fixed random seeds for fair comparison.

To execute the Object-Oriented pipeline:
```bash
python oop/run.py
```
To execute the Functional Programming pipeline:
```bash
python fp/run.py
```

_Note: Execution will automatically generate fitness evolution plots (*oop.png and *fp.png) to prevent file overwriting, alongside detailed performance logs (.json) in the reports/ directory._

### Running Unit Tests
The project features strict test coverage for selection, crossover, mutation, and generational improvement.
```bash
python -m unittest discover -s oop/tests -t oop
python -m unittest discover -s fp/tests -t fp
```
## 4. Reflection: OOP vs. FP Trade-offs
Transitioning the GA engine between OOP and FP paradigms revealed significant architectural and computational trade-offs, perfectly reflecting the empirical runtime logs:

- **Object-Oriented Programming (OOP):** The OOP implementation utilizes the Strategy Pattern to decouple genetic operators from the core engine. Modeling biological processes natively aligns with OOP; a `Chromosome` object "mutates" by altering its internal state in place. Because Python lists are mutable, in-place bit-flipping involves negligible memory allocation overhead. This resulted in superior raw execution speed. However, state mutability introduces risk: strict encapsulation (using `@property` and `setters`) is mandatorily required to manually invalidate and reset the cached `_fitness` whenever a chromosome's genes are altered, otherwise, the algorithm will suffer from stale state bugs.

- **Functional Programming (FP):** The FP pipeline discards classes and mutable states, relying strictly on pure functions (`map`, `reduce`, `filter`), closures, and immutable `tuples`. Generational evolution is achieved by recursively folding states rather than via iterative `for` loops. While this design inherently eliminates side-effects—making the codebase robust and natively thread-safe for distributed computing—it introduces a noticeable performance penalty. Python's garbage collection struggles with the continuous memory allocation required to instantiate entirely new tuples for thousands of offspring in every generation, causing the FP execution time to be noticeably slower than OOP.

<p align="center">
  <img src="reports/knapsack_curve_oop.png" width="45%" />
  <img src="reports/knapsack_curve_fp.png" width="45%" />
</p>
<p align="center">
  <em>Comparison of knapsack problem: OOP vs Functional Programming</em>
</p>

Ultimately, OOP proved computationally superior for the raw iterative speed required by localized heuristic searches, whereas FP enforced a highly scalable, safe, and side-effect-free data transformation architecture.
## 5. Extra design: GA in Feature Selection
In the realm of machine learning, training an algorithm with hundreds of features often leads to the curse of dimensionality and severe overfitting. Exhaustive search for the optimal subset of 100 features yields $2^{100}$ combinations, which is computationally impossible. GA acts as a highly efficient directed global search mechanism for this binary space.

By successfully extending the architecture to solve the **Feature Selection** problem:

1. **Simulated Landscape & L0 Penalty:** The current implementation uses a stochastic simulation environment (`random.uniform`) to inject 20 highly informative signals amidst 80 noise variables. The true power of the GA lies in its customized objective function. It calculates the dot product of the binary chromosome and the importance scores, but crucially, it subtracts a penalty for every feature included (L0 Regularization). The GA autonomously learns to drop a feature if its predictive contribution is too marginal to justify the complexity penalty.
2. **Architectural Extensibility:** The problem is intentionally designed as a "Mock" environment. The core Genetic Algorithm engine (`ga.py`) is completely decoupled from the problem domain. As demonstrated in the source code comments, this simulation can be instantly replaced by a real-world pipeline (e.g., using `scikit-learn`'s `RandomForestClassifier` to extract exact `feature_importances_`) without modifying a single line of the underlying evolutionary logic. 

This design proves that the implemented GA framework is highly scalable, robust, and mathematically ready to be deployed as an automatic feature selection pipeline in real-world Data Science projects.
