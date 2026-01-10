# Sundar-Tibshirani Gap Statistic

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn compatible](https://img.shields.io/badge/scikit--learn-compatible-orange.svg)](https://scikit-learn.org/)

A generalized Gap Statistic implementation for evaluating **any** cluster solution, not just k-means.

## Overview

The original Gap Statistic (Tibshirani, Walther, & Hastie, 2001) is a popular method for determining the optimal number of clusters. However, it was designed specifically to evaluate k-means clustering solutions generated during its own optimization process.

The **Sundar-Tibshirani Gap Statistic** extends this framework to evaluate arbitrary cluster solutions from:

- Hierarchical clustering
- DBSCAN and density-based methods
- Gaussian Mixture Models
- Spectral clustering
- Expert-defined segments
- Any other clustering approach

## Key Innovation

| Original Gap Statistic | Sundar-Tibshirani Gap Statistic |
|------------------------|----------------------------------|
| Clusters reference data with k-means | Applies user-provided labels to reference data |
| Algorithm-specific | Algorithm-agnostic |
| Evaluates k-means solutions only | Evaluates **any** cluster assignment |

## Installation

```bash
pip install sundar-gap-stat
```

Or install from source:

```bash
git clone https://github.com/pvsundar/sundar_gap_stat.git
cd sundar_gap_stat
pip install -e .
```

## Quick Start

```python
from sundar_gap_stat import SundarTibshiraniGapStatistic
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np

# Your data
X = np.random.randn(200, 4)
X_scaled = StandardScaler().fit_transform(X)

# Any clustering algorithm
agg = AgglomerativeClustering(n_clusters=3)
labels = agg.fit_predict(X_scaled)

# Evaluate with Sundar-Tibshirani Gap Statistic
gap_stat = SundarTibshiraniGapStatistic(
    pca_sampling=True,
    use_user_labels=True,  # Key parameter!
    return_params=True
)

gap_value, params = gap_stat.compute_gap_statistic(
    X=X_scaled,
    labels=labels,
    B=100  # Number of reference samples
)

print(f"Gap Statistic: {gap_value:.3f}")
print(f"Standard Error: {params['sim_sks']:.3f}")
```

## Comparing Multiple Clustering Solutions

```python
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Initialize Gap Statistic
gap_stat = SundarTibshiraniGapStatistic(use_user_labels=True)

# Compare different algorithms
algorithms = {
    'K-Means': KMeans(n_clusters=3, n_init=15, random_state=42),
    'Agglomerative': AgglomerativeClustering(n_clusters=3),
    'GMM': GaussianMixture(n_components=3, random_state=42)
}

for name, algo in algorithms.items():
    labels = algo.fit_predict(X_scaled)
    gap = gap_stat.compute_gap_statistic(X_scaled, labels, B=100)
    print(f"{name}: Gap = {gap:.3f}")
```

## Finding Optimal k with Any Algorithm

```python
# Use hierarchical clustering to find optimal k
gaps = []
for k in range(2, 8):
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(X_scaled)
    gap = gap_stat.compute_gap_statistic(X_scaled, labels, B=100)
    gaps.append((k, gap))
    print(f"k={k}: Gap = {gap:.3f}")

# Select k using elbow method or standard Gap criterion
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distance_metric` | str or callable | 'euclidean' | Distance metric for dispersion calculation |
| `pca_sampling` | bool | True | Use PCA-based reference distribution (recommended) |
| `standardize_within_pca` | bool | False | Standardize before PCA |
| `use_user_labels` | bool | True | **True** = Sundar-Tibshirani extension; **False** = original Tibshirani |
| `return_params` | bool | False | Return additional diagnostics |
| `n_init` | int | 12 | K-means initializations (only if `use_user_labels=False`) |
| `random_state` | int | 7142 | Random seed for reproducibility |

## Returned Parameters

When `return_params=True`, the method returns a tuple `(gap, params)` where `params` contains:

- `Wk`: Observed within-cluster dispersion
- `sim_Wks`: Array of simulated Wk values from reference distributions
- `sim_sks`: Adjusted standard error (sqrt(1 + 1/B) * SD)
- `gap`: Gap statistic value
- `sd_k`: Standard deviation of log(sim_Wks)

## Mathematical Foundation

The Sundar-Tibshirani Gap Statistic is defined as:

$$\text{Gap}_{\text{ST}}(k) = E^*[\log(W_k^*(\mathbf{L}))] - \log(W_k(\mathbf{L}))$$

where:
- $\mathbf{L}$ = fixed cluster labels from any source
- $W_k^*(\mathbf{L})$ = within-cluster dispersion applying labels $\mathbf{L}$ to reference data
- $E^*$ = expectation over reference distributions

This differs from the original Gap Statistic, which re-clusters reference data with k-means.

## Simulation Study Results

Comprehensive simulations demonstrate:

1. **Correct k detection** across well-separated, overlapping, and elongated cluster structures
2. **Algorithm invariance**: Consistent evaluation across K-Means, Agglomerative, and GMM
3. **Noise robustness**: Maintains stability under moderate noise better than Silhouette
4. **Sample requirements**: Reliable estimates with n >= 100 observations
5. **Monte Carlo convergence**: B = 100 provides excellent precision

## Citation

If you use this package in your research, please cite:

```bibtex
@article{balakrishnan2026sundar,
  title={The Sundar-Tibshirani Gap Statistic: A Goodness-of-Fit Measure 
         for Evaluating Any Cluster Solution},
  author={Balakrishnan, P. V. Sundar},
  journal={Journal of Marketing Analytics},
  year={2026}
}
```

## References

- Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters in a data set via the gap statistic. *Journal of the Royal Statistical Society: Series B*, 63(2), 411-423.

## License

MIT License - see [LICENSE](LICENSE) file.

## Author

**P. V. Sundar Balakrishnan**  
University of Washington Bothell  
School of Business  
sundar@uw.edu
