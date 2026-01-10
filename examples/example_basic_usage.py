"""
Example 1: Basic Usage of Sundar-Tibshirani Gap Statistic
=========================================================

This example demonstrates the basic usage of the Gap Statistic
for evaluating cluster solutions from different algorithms.
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sundar_gap_stat import SundarTibshiraniGapStatistic, gap_statistic

# Generate sample data with 3 true clusters
print("Generating sample data...")
X, y_true = make_blobs(
    n_samples=300, 
    n_features=4, 
    centers=3, 
    cluster_std=1.0, 
    random_state=42
)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Data shape: {X_scaled.shape}")
print(f"True number of clusters: {len(np.unique(y_true))}")
print()

# =============================================================================
# Method 1: Using the convenience function
# =============================================================================
print("=" * 60)
print("METHOD 1: Using gap_statistic() convenience function")
print("=" * 60)

# Cluster with K-Means
kmeans = KMeans(n_clusters=3, n_init=15, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Compute Gap Statistic
gap, se = gap_statistic(X_scaled, labels_kmeans, B=100)
print(f"K-Means (k=3): Gap = {gap:.3f}, SE = {se:.3f}")
print()

# =============================================================================
# Method 2: Using the class directly with return_params
# =============================================================================
print("=" * 60)
print("METHOD 2: Using SundarTibshiraniGapStatistic class")
print("=" * 60)

# Initialize the Gap Statistic calculator
gap_stat = SundarTibshiraniGapStatistic(
    pca_sampling=True,
    use_user_labels=True,  # Key: This makes it Sundar-Tibshirani extension
    return_params=True
)

# Evaluate true labels
gap_true, params_true = gap_stat.compute_gap_statistic(
    X_scaled, y_true, B=100
)
print(f"True Labels (k=3): Gap = {gap_true:.3f}, SE = {params_true['sim_sks']:.3f}")
print(f"  - Observed Wk: {params_true['Wk']:.4f}")
print(f"  - Mean simulated Wk: {np.mean(params_true['sim_Wks']):.4f}")
print()

# =============================================================================
# Compare different clustering algorithms
# =============================================================================
print("=" * 60)
print("COMPARING CLUSTERING ALGORITHMS")
print("=" * 60)

algorithms = {
    'K-Means': KMeans(n_clusters=3, n_init=15, random_state=42),
    'Agglomerative (Ward)': AgglomerativeClustering(n_clusters=3, linkage='ward'),
    'Agglomerative (Complete)': AgglomerativeClustering(n_clusters=3, linkage='complete'),
    'GMM': GaussianMixture(n_components=3, random_state=42),
}

results = []
for name, algo in algorithms.items():
    labels = algo.fit_predict(X_scaled)
    gap, params = gap_stat.compute_gap_statistic(X_scaled, labels, B=100)
    results.append((name, gap, params['sim_sks']))
    print(f"{name}: Gap = {gap:.3f} (SE = {params['sim_sks']:.3f})")

print()

# =============================================================================
# Find optimal k using Gap criterion
# =============================================================================
print("=" * 60)
print("FINDING OPTIMAL K USING GAP CRITERION")
print("=" * 60)

print("\nK-Means across k=2 to k=7:")
print("-" * 40)

gaps_by_k = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=15, random_state=42)
    labels = km.fit_predict(X_scaled)
    gap, params = gap_stat.compute_gap_statistic(X_scaled, labels, B=100)
    gaps_by_k.append({'k': k, 'gap': gap, 'se': params['sim_sks']})
    print(f"k={k}: Gap = {gap:.3f} (SE = {params['sim_sks']:.3f})")

# Apply Gap criterion: first k where Gap(k) >= Gap(k+1) - SE(k+1)
print("\nApplying Gap criterion:")
for i in range(len(gaps_by_k) - 1):
    current = gaps_by_k[i]
    next_val = gaps_by_k[i + 1]
    threshold = next_val['gap'] - next_val['se']
    satisfied = current['gap'] >= threshold
    print(f"k={current['k']}: Gap({current['k']}) = {current['gap']:.3f} >= "
          f"Gap({next_val['k']}) - SE = {threshold:.3f} --> {satisfied}")
    if satisfied:
        print(f"\n*** Optimal k = {current['k']} ***")
        break

print("\nDone!")
