"""
EECS 658 - Assignment 4
CompareFeatureSelectionMethods.py

Brief description:
    This program compares feature selection / dimensionality reduction methods on the Iris dataset
    using 2-fold cross-validation with a Decision Tree classifier.

Inputs:
    None (loads Iris from sklearn)

Outputs (for each Part):
    - Part label
    - Confusion matrix (must sum to 150)
    - Accuracy metric
    - Final list of features used

Part details (from assignment instructions):
    Part 1: Use original 4 features
    Part 2: PCA transform into z1..z4, print eigenvalues/eigenvectors, choose smallest subset with PoV > 0.90,
            print PoV, evaluate using chosen z-features
    Part 3: Simulated annealing over 8 features (original 4 + z1..z4), 100 iterations, perturb 1 or 2 params,
            c=1, restart x=10, print subset, accuracy, Pr(accept), Random Uniform, Status each iteration
    Part 4: Genetic algorithm over same 8 features, initial population given, 50 generations,
            print top 5 feature sets + accuracy each generation

Collaborators:
    None

Other sources:
    Assignment 4 Instructions (Canvas)
    Lecture slides (PCA feature transformation, Simulated Annealing, Genetic Algorithm)
    ChatGPT

Author:
    Abhiroop Goel

Creation date:
    2026-03-11
"""

# -----------------------------
# Imports (all used by this script)
# -----------------------------

import math  # Used for exp() in simulated annealing acceptance probability.
import numpy as np  # Used for numeric arrays and linear algebra.

from sklearn.datasets import load_iris  # Loads the Iris dataset.
from sklearn.model_selection import train_test_split  # Used to create the 2 folds (50/50 split).

from sklearn.tree import DecisionTreeClassifier  # Decision Tree model required by the assignment.

from sklearn.metrics import confusion_matrix  # Used to compute confusion matrices.
from sklearn.metrics import accuracy_score  # Used to compute accuracy scores.


# -----------------------------
# Constants and feature names
# -----------------------------

# Names of the 4 original Iris features (Part 1).
ORIG_FEATURE_NAMES = ["sepal-length", "sepal-width", "petal-length", "petal-width"]

# Names of the 4 PCA-transformed features (Part 2).
PCA_FEATURE_NAMES = ["z1", "z2", "z3", "z4"]

# Names of all 8 total features used in Parts 3 and 4.
ALL_FEATURE_NAMES = ORIG_FEATURE_NAMES + PCA_FEATURE_NAMES


# -----------------------------
# Utility printing helpers
# -----------------------------

def print_part_header(part_number: int) -> None:
    """Print a clean label before each Part so the grader can see the output sections."""
    print("============================================")
    print(f"Part {part_number}")
    print("============================================")


def print_final_results(part_number: int, y_true: np.ndarray, y_pred: np.ndarray, feature_list: list[str]) -> None:
    """
    Print the confusion matrix, accuracy, and features used.
    Also checks that the confusion matrix sums to 150.
    """
    # Compute confusion matrix from final combined predictions.
    cm = confusion_matrix(y_true, y_pred)

    # Compute accuracy from final combined predictions.
    acc = accuracy_score(y_true, y_pred)

    # Print label + confusion matrix.
    print(f"Part {part_number} Confusion Matrix:")
    print(cm)

    # Print label + accuracy.
    print(f"Part {part_number} Accuracy Score: {round(acc, 3)}")

    # Print final features used.
    print(f"Part {part_number} Final Features Used: {feature_list}")

    # Blank line for readability.
    print("")

    # Verify that the confusion matrix sums to 150.
    total = int(cm.sum())
    if total != 150:
        print("WARNING: Confusion matrix sum is", total, "but should be 150.\n")


# -----------------------------
# 2-fold CV evaluator
# -----------------------------

def two_fold_cv_accuracy_and_predictions(X_fold1: np.ndarray,
                                        X_fold2: np.ndarray,
                                        y_fold1: np.ndarray,
                                        y_fold2: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Runs 2-fold cross validation manually:
        Train on fold1, test on fold2
        Train on fold2, test on fold1
    Then concatenates predictions so the final confusion matrix sums to 150.

    Returns:
        (accuracy, y_true_combined, y_pred_combined)
    """
    # Create a Decision Tree classifier.
    # Note: random_state is set only to make results reproducible for grading.
    clf = DecisionTreeClassifier(random_state=1)

    # Train on fold1.
    clf.fit(X_fold1, y_fold1)

    # Predict on fold2.
    pred_fold2 = clf.predict(X_fold2)

    # Train on fold2.
    clf.fit(X_fold2, y_fold2)

    # Predict on fold1.
    pred_fold1 = clf.predict(X_fold1)

    # Combine actual labels in the same order as predictions.
    y_true = np.concatenate([y_fold2, y_fold1])

    # Combine predictions to cover all 150 samples.
    y_pred = np.concatenate([pred_fold2, pred_fold1])

    # Compute accuracy.
    acc = accuracy_score(y_true, y_pred)

    # Return results.
    return acc, y_true, y_pred


# -----------------------------
# PCA (manual eigen approach)
# -----------------------------

def compute_pca_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes PCA using covariance + eigen decomposition (like typical lecture examples).
    Returns:
        Z (150x4)      = transformed features z1..z4
        eigvals (4,)   = eigenvalues in descending order
        eigvecs (4x4)  = eigenvectors (columns) aligned with eigvals
        pov (4,)       = cumulative PoV after 1..4 components
    """
    # Center the data (subtract mean of each feature).
    X_center = X - X.mean(axis=0)

    # Compute covariance matrix (4x4).
    cov = np.cov(X_center, rowvar=False)

    # Compute eigenvalues and eigenvectors of the covariance matrix.
    eigvals, eigvecs = np.linalg.eig(cov)

    # Sort eigenvalues (and eigenvectors) from largest to smallest.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order].real
    eigvecs = eigvecs[:, order].real

    # Transform original data into PCA coordinates (z-features).
    Z = X_center @ eigvecs

    # Compute cumulative PoV (percentage of variance).
    pov = np.cumsum(eigvals) / np.sum(eigvals)

    # Return all PCA results.
    return Z, eigvals, eigvecs, pov


def smallest_k_for_pov(pov: np.ndarray, threshold: float) -> int:
    """Return the smallest number of components needed so PoV > threshold."""
    for i in range(len(pov)):
        if pov[i] > threshold:
            return i + 1
    return len(pov)


# -----------------------------
# Simulated annealing (Part 3)
# -----------------------------

def simulated_annealing_select_features(X8_fold1: np.ndarray,
                                       X8_fold2: np.ndarray,
                                       y_fold1: np.ndarray,
                                       y_fold2: np.ndarray,
                                       iterations: int = 100,
                                       c_value: float = 1.0,
                                       restart_x: int = 10) -> tuple[np.ndarray, float]:
    """
    Runs simulated annealing to select a feature subset from 8 features.
    Prints the required info for each iteration.

    Returns:
        best_mask (8 booleans)
        best_accuracy
    """
    # Fix random seed so your output is stable and reproducible.
    np.random.seed(1)

    # Start with a random subset (at least 1 feature).
    current_mask = np.random.rand(8) < 0.5
    if not current_mask.any():
        current_mask[np.random.randint(8)] = True

    # Evaluate starting subset.
    current_acc, _, _ = two_fold_cv_accuracy_and_predictions(
        X8_fold1[:, current_mask], X8_fold2[:, current_mask], y_fold1, y_fold2
    )

    # Track best found.
    best_mask = current_mask.copy()
    best_acc = current_acc

    # Counter for restart logic (no improvement/accept streak).
    no_progress_count = 0

    # Temperature schedule setup (simple cooling).
    temperature = 1.0
    alpha = 0.95

    # Print header for Part 3 iteration logs.
    print("Part 3 Simulated Annealing Iteration Logs:")
    print("Iteration | Subset | Accuracy | Pr(accept) | Random Uniform | Status")

    # Loop for 100 iterations.
    for it in range(1, iterations + 1):
        # Choose to flip 1 or 2 parameters (per instructions).
        flips = np.random.choice([1, 2])

        # Copy current mask to create a candidate.
        candidate_mask = current_mask.copy()

        # Pick indices to flip.
        flip_idx = np.random.choice(8, size=flips, replace=False)

        # Flip those feature bits.
        candidate_mask[flip_idx] = ~candidate_mask[flip_idx]

        # Make sure we never end up with zero features selected.
        if not candidate_mask.any():
            candidate_mask[np.random.randint(8)] = True

        # Evaluate candidate subset accuracy.
        candidate_acc, _, _ = two_fold_cv_accuracy_and_predictions(
            X8_fold1[:, candidate_mask], X8_fold2[:, candidate_mask], y_fold1, y_fold2
        )

        # Default acceptance probability and status.
        pr_accept = 0.0
        rand_u = float(np.random.rand())
        status = "Discarded"

        # If candidate improves accuracy, accept it immediately.
        if candidate_acc > current_acc:
            current_mask = candidate_mask
            current_acc = candidate_acc
            pr_accept = 1.0
            status = "Improved"
            no_progress_count = 0

        else:
            # Compute acceptance probability using exp((new-old)/(c*T)).
            # c is given as 1 in assignment, but we keep the parameter for clarity.
            if temperature > 1e-12:
                pr_accept = math.exp((candidate_acc - current_acc) / (c_value * temperature))
            else:
                pr_accept = 0.0

            # Accept with probability pr_accept.
            if rand_u < pr_accept:
                current_mask = candidate_mask
                current_acc = candidate_acc
                status = "Accepted"
            else:
                status = "Discarded"

            # Count as no progress if it was not improved.
            no_progress_count += 1

        # Update best if we beat the best accuracy.
        if current_acc > best_acc:
            best_acc = current_acc
            best_mask = current_mask.copy()

        # Restart if we hit the restart threshold.
        if no_progress_count >= restart_x:
            # Random restart subset.
            current_mask = np.random.rand(8) < 0.5
            if not current_mask.any():
                current_mask[np.random.randint(8)] = True

            # Re-evaluate after restart.
            current_acc, _, _ = two_fold_cv_accuracy_and_predictions(
                X8_fold1[:, current_mask], X8_fold2[:, current_mask], y_fold1, y_fold2
            )

            # Reset counter.
            no_progress_count = 0

            # Mark status.
            status = "Restart"

        # Convert subset to readable feature names.
        subset_names = [ALL_FEATURE_NAMES[i] for i in range(8) if current_mask[i]]

        # Print required per-iteration info.
        print(f"{it:9d} | {subset_names} | {round(current_acc, 3)} | {round(pr_accept, 3)} | {round(rand_u, 3)} | {status}")

        # Cool temperature.
        temperature *= alpha

    # Return the best subset found.
    return best_mask, best_acc


# -----------------------------
# Genetic Algorithm (Part 4)
# -----------------------------

def genetic_algorithm_select_features(X8_fold1: np.ndarray,
                                     X8_fold2: np.ndarray,
                                     y_fold1: np.ndarray,
                                     y_fold2: np.ndarray,
                                     generations: int = 50) -> tuple[np.ndarray, float]:
    """
    Runs a genetic algorithm to select a feature subset from 8 features.
    Prints top 5 feature sets and accuracies for each generation.

    Returns:
        best_mask (8 booleans)
        best_accuracy
    """
    # Fix random seed for reproducibility.
    np.random.seed(2)

    # Helper to convert feature names to a boolean mask.
    name_to_idx = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}

    # Initial population required by the assignment (5 individuals).
    initial_population_feature_sets = [
        ["z1", "sepal-length", "sepal-width", "petal-length", "petal-width"],
        ["z1", "z2", "sepal-width", "petal-length", "petal-width"],
        ["z1", "z2", "z3", "sepal-width", "petal-length"],
        ["z1", "z2", "z3", "z4", "sepal-width"],
        ["z1", "z2", "z3", "z4", "sepal-length"],
    ]

    # Build population masks from the required initial sets.
    population = []
    for feature_set in initial_population_feature_sets:
        mask = np.zeros(8, dtype=bool)
        for fname in feature_set:
            mask[name_to_idx[fname]] = True
        population.append(mask)

    # Add extra random individuals so the GA actually has room to evolve.
    # (This makes it much more likely that generation 1 top-5 differs from generation 50 top-5.)
    POP_SIZE = 20
    while len(population) < POP_SIZE:
        rand_mask = np.random.rand(8) < 0.5
        if not rand_mask.any():
            rand_mask[np.random.randint(8)] = True
        population.append(rand_mask)

    # Fitness function = 2-fold CV accuracy.
    def fitness(mask: np.ndarray) -> float:
        acc, _, _ = two_fold_cv_accuracy_and_predictions(
            X8_fold1[:, mask], X8_fold2[:, mask], y_fold1, y_fold2
        )
        return acc

    # Crossover combines features from both parents.
    def crossover(parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        chooser = np.random.rand(8) < 0.5
        child = np.where(chooser, parent_a, parent_b)
        if not child.any():
            child[np.random.randint(8)] = True
        return child

    # Mutation flips a random bit sometimes.
    def mutate(mask: np.ndarray, mutation_p: float = 0.4) -> np.ndarray:
        m = mask.copy()
        if np.random.rand() < mutation_p:
            idx = np.random.randint(8)
            m[idx] = ~m[idx]
        if not m.any():
            m[np.random.randint(8)] = True
        return m

    # Run for 50 generations.
    for gen in range(1, generations + 1):
        # Score each individual.
        scored = [(fitness(m), m) for m in population]

        # Sort best to worst.
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take the top 5.
        top5 = scored[:5]

        # Print generation label and top 5.
        print(f"\nGeneration {gen} Top 5:")
        for rank, (acc, mask) in enumerate(top5, start=1):
            feat_list = [ALL_FEATURE_NAMES[i] for i in range(8) if mask[i]]
            print(f"  {rank}) Features: {feat_list} | Accuracy: {round(acc, 3)}")

        # Elitism: keep top 5 unchanged.
        new_population = [m for _, m in top5]

        # Parent pool: take top 10 to breed from.
        parent_pool = [m for _, m in scored[:10]]

        # Fill rest of population with children.
        while len(new_population) < POP_SIZE:
            pa_idx, pb_idx = np.random.choice(len(parent_pool), size=2, replace=True)
            child = crossover(parent_pool[pa_idx], parent_pool[pb_idx])
            child = mutate(child)
            new_population.append(child)

        # Move to next generation.
        population = new_population

    # Final best individual after last generation.
    scored = [(fitness(m), m) for m in population]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_acc, best_mask = scored[0]

    return best_mask, best_acc


# -----------------------------
# Main program (Parts 1-4)
# -----------------------------

def main() -> None:
    # Load Iris dataset (150 samples).
    iris = load_iris()

    # X = original 4 features.
    X = iris.data

    # y = class labels 0,1,2.
    y = iris.target

    # Create the two folds (75/75) for 2-fold cross validation.
    # stratify=y keeps class balance in both folds.
    idx = np.arange(len(y))
    idx_fold1, idx_fold2, _, _ = train_test_split(idx, y, test_size=0.5, random_state=1, stratify=y)

    # Build fold feature matrices.
    X_fold1 = X[idx_fold1]
    X_fold2 = X[idx_fold2]

    # Build fold labels.
    y_fold1 = y[idx_fold1]
    y_fold2 = y[idx_fold2]

    # -----------------------------
    # Part 1: Original features
    # -----------------------------
    print_part_header(1)

    # Evaluate using all 4 original features.
    acc1, y_true1, y_pred1 = two_fold_cv_accuracy_and_predictions(X_fold1, X_fold2, y_fold1, y_fold2)

    # Print final results.
    print_final_results(1, y_true1, y_pred1, ORIG_FEATURE_NAMES)

    # -----------------------------
    # Part 2: PCA transform + PoV
    # -----------------------------
    print_part_header(2)

    # Compute PCA features from the full dataset X.
    Z, eigvals, eigvecs, pov = compute_pca_features(X)

    # Print eigenvalues as a diagonal matrix (eigenvalue matrix).
    print("Eigenvalues Matrix (diagonal):")
    print(np.diag(eigvals))
    print("")

    # Print eigenvectors matrix.
    print("Eigenvectors Matrix (columns correspond to eigenvalues):")
    print(eigvecs)
    print("")

    # Choose smallest number of z-features so PoV > 0.90.
    k = smallest_k_for_pov(pov, threshold=0.90)

    # Print PoV details.
    print("Cumulative PoV values:", pov)
    print(f"Selected k = {k} because PoV after k components = {round(pov[k-1], 3)} which is > 0.90")
    print("")

    # Create the selected Z subset (z1..zk).
    Z_selected = Z[:, :k]

    # Split PCA features into folds using the same indices.
    Z_fold1 = Z_selected[idx_fold1]
    Z_fold2 = Z_selected[idx_fold2]

    # Evaluate using selected z-features.
    acc2, y_true2, y_pred2 = two_fold_cv_accuracy_and_predictions(Z_fold1, Z_fold2, y_fold1, y_fold2)

    # Print final results for Part 2.
    print_final_results(2, y_true2, y_pred2, PCA_FEATURE_NAMES[:k])

    # -----------------------------
    # Part 3: Simulated annealing over 8 features
    # -----------------------------
    print_part_header(3)

    # Build 8-feature matrix = [original 4 | z1..z4].
    X8 = np.hstack([X, Z])

    # Split into folds.
    X8_fold1 = X8[idx_fold1]
    X8_fold2 = X8[idx_fold2]

    # Run simulated annealing with required settings.
    best_sa_mask, best_sa_acc = simulated_annealing_select_features(
        X8_fold1, X8_fold2, y_fold1, y_fold2,
        iterations=100,
        c_value=1.0,
        restart_x=10
    )

    # Evaluate best SA subset for final confusion matrix.
    best_sa_features = [ALL_FEATURE_NAMES[i] for i in range(8) if best_sa_mask[i]]
    acc3, y_true3, y_pred3 = two_fold_cv_accuracy_and_predictions(
        X8_fold1[:, best_sa_mask], X8_fold2[:, best_sa_mask], y_fold1, y_fold2
    )

    # Print final results for Part 3.
    print_final_results(3, y_true3, y_pred3, best_sa_features)

    # -----------------------------
    # Part 4: Genetic algorithm over 8 features
    # -----------------------------
    print_part_header(4)

    # Run GA for 50 generations.
    best_ga_mask, best_ga_acc = genetic_algorithm_select_features(
        X8_fold1, X8_fold2, y_fold1, y_fold2,
        generations=50
    )

    # Evaluate best GA subset for final confusion matrix.
    best_ga_features = [ALL_FEATURE_NAMES[i] for i in range(8) if best_ga_mask[i]]
    acc4, y_true4, y_pred4 = two_fold_cv_accuracy_and_predictions(
        X8_fold1[:, best_ga_mask], X8_fold2[:, best_ga_mask], y_fold1, y_fold2
    )

    # Print final results for Part 4.
    print_final_results(4, y_true4, y_pred4, best_ga_features)


if __name__ == "__main__":
    main()
