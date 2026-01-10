"""
Sundar-Tibshirani Gap Statistic
===============================

A generalized Gap Statistic implementation for evaluating any cluster solution.

The original Gap Statistic (Tibshirani, Walther, & Hastie, 2001) was designed
specifically for k-means clustering. The Sundar-Tibshirani extension allows
evaluation of arbitrary cluster solutions from any algorithm.

Author: P.V. Sundar Balakrishnan
Email: sundar@uw.edu
License: MIT
"""

import numpy as np
from typing import Callable, Union, Tuple, Optional
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
from sklearn.cluster import KMeans

__version__ = "2.0.0"
__author__ = "P.V. (Sundar) Balakrishnan"


class SundarTibshiraniGapStatistic:
    """
    Implements the Sundar-Tibshirani Gap Statistic for cluster analysis.
    
    This class provides methods to calculate the Gap Statistic for a given
    cluster assignment, enabling evaluation of any clustering solution
    regardless of the algorithm that produced it.
    
    Parameters
    ----------
    distance_metric : str or callable, default='euclidean'
        The distance metric for computing within-cluster dispersion.
        Valid string options: 'euclidean', 'manhattan', 'cosine', 
        'minkowski', 'l1', 'l2'.
        Can also be a callable function that takes two arrays and returns
        a distance matrix.
        
    pca_sampling : bool, default=True
        Whether to use PCA-based reference distribution sampling.
        When True, reference data is generated in the principal component
        space, which better preserves the correlation structure of the
        original data. Recommended for most applications.
        
    standardize_within_pca : bool, default=False
        Whether to standardize data before computing PCA for reference
        sampling. Usually not necessary if data is already standardized.
        
    return_params : bool, default=False
        Whether to return additional diagnostic parameters including
        the observed Wk, simulated Wk values, and standard errors.
        
    use_user_labels : bool, default=True
        The key parameter distinguishing Sundar-Tibshirani from original.
        
        - True (Sundar-Tibshirani): Applies user-provided cluster labels
          to reference data. Enables evaluation of ANY clustering solution.
        
        - False (Original Tibshirani): Clusters reference data with k-means.
          Only valid for evaluating k-means solutions.
          
    n_init : int, default=12
        Number of k-means initializations. Only used when use_user_labels=False.
        
    random_state : int, default=7142
        Random seed for reproducibility.
        
    Attributes
    ----------
    valid_metrics : list
        List of valid string distance metric names.
        
    Examples
    --------
    >>> from sundar_gap_stat import SundarTibshiraniGapStatistic
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> import numpy as np
    >>> 
    >>> # Generate sample data
    >>> X = np.random.randn(200, 4)
    >>> 
    >>> # Any clustering algorithm
    >>> agg = AgglomerativeClustering(n_clusters=3)
    >>> labels = agg.fit_predict(X)
    >>> 
    >>> # Compute Gap Statistic
    >>> gap_stat = SundarTibshiraniGapStatistic(use_user_labels=True)
    >>> gap = gap_stat.compute_gap_statistic(X, labels, B=100)
    >>> print(f"Gap: {gap:.3f}")
    
    Notes
    -----
    The Sundar-Tibshirani Gap Statistic is defined as:
    
    Gap_ST(k) = E*[log(W_k*(L))] - log(W_k(L))
    
    where L is the fixed cluster label vector and W_k*(L) is the 
    within-cluster dispersion computed by applying labels L to
    reference data.
    
    References
    ----------
    Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the 
    number of clusters in a data set via the gap statistic. Journal of 
    the Royal Statistical Society: Series B, 63(2), 411-423.
    """

    def __init__(
        self,
        distance_metric: Union[str, Callable] = 'euclidean',
        pca_sampling: bool = True,
        standardize_within_pca: bool = False,
        return_params: bool = False,
        use_user_labels: bool = True,
        n_init: int = 12,
        random_state: int = 7142
    ):
        self.valid_metrics = ['euclidean', 'minkowski', 'manhattan', 
                              'cosine', 'l1', 'l2']
        
        if isinstance(distance_metric, str):
            if distance_metric not in self.valid_metrics:
                raise ValueError(
                    f"Invalid distance metric '{distance_metric}'. "
                    f"Choose from {self.valid_metrics}"
                )
            self.distance_metric = lambda X, Y: pairwise_distances(
                X, Y, metric=distance_metric
            )
            self._metric_name = distance_metric
        elif callable(distance_metric):
            self.distance_metric = distance_metric
            self._metric_name = 'custom'
        else:
            raise TypeError(
                "distance_metric must be either a string or a callable function"
            )
        
        self.pca_sampling = pca_sampling
        self.standardize_within_pca = standardize_within_pca
        self.return_params = return_params
        self.use_user_labels = use_user_labels
        self.n_init = n_init
        self.random_state = random_state

    def _rng(self) -> np.random.Generator:
        """Return a fresh RNG seeded from self.random_state for deterministic runs."""
        return np.random.default_rng(self.random_state)

    def _calculate_within_cluster_dispersion(
        self, 
        X: np.ndarray, 
        labels: np.ndarray
    ) -> float:
        """
        Calculate pooled within-cluster dispersion Wk.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).
        labels : np.ndarray
            Cluster labels of shape (n_samples,).
            
        Returns
        -------
        float
            The within-cluster dispersion value.
        """
        unique_labels = np.unique(labels)
        
        # Compute cluster centroids
        centroids = np.array([
            X[labels == lab].mean(axis=0) for lab in unique_labels
        ])

        Wk = 0.0
        for idx, lab in enumerate(unique_labels):
            cluster_points = X[labels == lab]
            if len(cluster_points) > 0:
                distances = self.distance_metric(
                    cluster_points, 
                    centroids[idx].reshape(1, -1)
                )
                Wk += np.sum(distances)

        # Normalize by 2n as per original Gap Statistic formulation
        Wk /= (2 * len(X))
        return Wk

    def _simulate_within_cluster_dispersions(
        self, 
        X: np.ndarray, 
        labels: np.ndarray, 
        k: int, 
        B: int,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Simulate Wk values for B reference distributions.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).
        labels : np.ndarray
            Cluster labels of shape (n_samples,).
        k : int
            Number of clusters.
        B : int
            Number of reference samples to generate.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
            
        Returns
        -------
        np.ndarray
            Array of B simulated Wk values.
        """
        n_samples, n_features = X.shape
        
        if rng is None:
            rng = self._rng()

        if self.pca_sampling:
            # Transform to principal component space
            if self.standardize_within_pca:
                scaler = StandardScaler()
                scaled_X = scaler.fit_transform(X)
            else:
                scaled_X = X
            
            # Compute SVD
            _, _, VT = svd(scaled_X, full_matrices=False)
            X_prime = np.dot(X, VT.T)
        else:
            X_prime = X
            VT = None

        simulated_Wks = []

        for _ in range(B):
            # Generate uniform reference data in transformed space
            Z_prime = rng.uniform(
                low=np.min(X_prime, axis=0), 
                high=np.max(X_prime, axis=0), 
                size=(n_samples, n_features)
            )

            # Transform back to original space if using PCA
            if self.pca_sampling and VT is not None:
                sampled_X = np.dot(Z_prime, VT)
            else:
                sampled_X = Z_prime

            # Key difference: use_user_labels controls behavior
            if self.use_user_labels:
                # SUNDAR-TIBSHIRANI: Apply user-provided labels to reference data
                Wk_star = self._calculate_within_cluster_dispersion(
                    sampled_X, labels
                )
            else:
                # ORIGINAL TIBSHIRANI: Cluster reference data with k-means
                kmeans = KMeans(
                    n_clusters=k, 
                    n_init=self.n_init, 
                    random_state=self.random_state
                )
                kmeans.fit(sampled_X)
                Wk_star = self._calculate_within_cluster_dispersion(
                    sampled_X, kmeans.labels_
                )

            simulated_Wks.append(Wk_star)

        return np.array(simulated_Wks)

    def compute_gap_statistic(
        self, 
        X: np.ndarray, 
        labels: np.ndarray, 
        k: Optional[int] = None, 
        B: int = 50
    ) -> Union[float, Tuple[float, dict]]:
        """
        Compute the Gap Statistic for a given cluster assignment.

        Parameters
        ----------
        X : np.ndarray or list
            Input data matrix of shape (n_samples, n_features).
            
        labels : np.ndarray
            Cluster labels of shape (n_samples,).
            
        k : int, optional
            Number of clusters. If None, inferred from unique values in labels.
            
        B : int, default=50
            Number of reference samples for Monte Carlo estimation.
            Recommended values: 50-100 for exploratory analysis,
            100-200 for final results.

        Returns
        -------
        gap : float
            The computed Gap Statistic value. Higher values indicate
            better clustering structure relative to random expectation.
            
        params : dict (only if return_params=True)
            Dictionary containing diagnostic parameters:
            
            - 'Wk': Observed within-cluster dispersion
            - 'sim_Wks': Array of simulated Wk values from reference data
            - 'sim_sks': Adjusted standard error = sqrt(1 + 1/B) * SD
            - 'gap': Gap Statistic value (same as returned gap)
            - 'sd_k': Standard deviation of log(sim_Wks)
            
        Raises
        ------
        TypeError
            If B is not an integer or X is not array-like.
        ValueError
            If B exceeds 500 (computational safeguard).
            
        Examples
        --------
        >>> gap_stat = SundarTibshiraniGapStatistic(return_params=True)
        >>> gap, params = gap_stat.compute_gap_statistic(X, labels, B=100)
        >>> print(f"Gap: {gap:.3f} (SE: {params['sim_sks']:.3f})")
        """
        # Input validation
        if not isinstance(B, int):
            raise TypeError('B must be of type int')
        if B > 500:
            raise ValueError(
                'B is too large (max 500). For most applications, B=100 '
                'provides sufficient precision.'
            )
        if B < 10:
            import warnings
            warnings.warn(
                f"B={B} is very small. Results may be unstable. "
                "Consider using B >= 50.",
                UserWarning
            )
            
        # Convert list to array if needed
        if isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray):
            raise TypeError(
                'X must be a numpy array or list, '
                f'got {type(X).__name__}'
            )
        
        # Infer k from labels if not provided
        if k is None:
            k = len(np.unique(labels))

        # Compute observed within-cluster dispersion
        Wk = self._calculate_within_cluster_dispersion(X, labels)
        
        # Simulate reference distribution with seeded RNG
        rng = self._rng()
        sim_Wks = self._simulate_within_cluster_dispersions(X, labels, k, B, rng=rng)

        # Compute Gap Statistic
        log_Wk = np.log(Wk)
        log_sim_Wks = np.log(sim_Wks)

        gap = np.mean(log_sim_Wks) - log_Wk
        sd_k = np.std(log_sim_Wks)
        sim_sks = np.sqrt(1 + (1 / B)) * sd_k  # Adjusted standard error

        if self.return_params:
            params = {
                'Wk': Wk, 
                'sim_Wks': sim_Wks, 
                'sim_sks': sim_sks, 
                'gap': gap, 
                'sd_k': sd_k
            }
            return gap, params
        else:
            return gap
    
    def __repr__(self):
        return (
            f"SundarTibshiraniGapStatistic("
            f"distance_metric='{self._metric_name}', "
            f"pca_sampling={self.pca_sampling}, "
            f"use_user_labels={self.use_user_labels})"
        )


def gap_statistic(
    X: np.ndarray,
    labels: np.ndarray,
    B: int = 100,
    pca_sampling: bool = True,
    use_user_labels: bool = True,
    random_state: int = 7142
) -> Tuple[float, float]:
    """
    Convenience function to compute Gap Statistic.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels of shape (n_samples,).
    B : int, default=100
        Number of reference samples.
    pca_sampling : bool, default=True
        Use PCA-based reference sampling.
    use_user_labels : bool, default=True
        True for Sundar-Tibshirani extension, False for original.
    random_state : int, default=7142
        Random seed.
        
    Returns
    -------
    gap : float
        Gap Statistic value.
    se : float
        Standard error.
        
    Examples
    --------
    >>> from sundar_gap_stat import gap_statistic
    >>> gap, se = gap_statistic(X, labels, B=100)
    >>> print(f"Gap: {gap:.3f} (SE: {se:.3f})")
    """
    gs = SundarTibshiraniGapStatistic(
        pca_sampling=pca_sampling,
        use_user_labels=use_user_labels,
        return_params=True,
        random_state=random_state
    )
    gap, params = gs.compute_gap_statistic(X, labels, B=B)
    return gap, params['sim_sks']


def find_optimal_k(
    X: np.ndarray,
    clustering_func: Callable,
    k_range: range = range(2, 11),
    B: int = 100,
    criterion: str = 'gap',
    pca_sampling: bool = True,
    random_state: int = 7142
) -> Tuple[int, dict]:
    """
    Find optimal number of clusters using Gap Statistic.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    clustering_func : callable
        Function that takes (X, k) and returns cluster labels.
        Example: lambda X, k: KMeans(n_clusters=k).fit_predict(X)
    k_range : range, default=range(2, 11)
        Range of k values to evaluate.
    B : int, default=100
        Number of reference samples.
    criterion : str, default='gap'
        Selection criterion:
        - 'gap': Standard Gap criterion (first k where Gap(k) >= Gap(k+1) - SE(k+1))
        - 'maxgap': Select k with maximum Gap value
        - 'elbow': Use second derivative to find elbow
    pca_sampling : bool, default=True
        Use PCA-based reference sampling.
    random_state : int, default=7142
        Random seed.
        
    Returns
    -------
    optimal_k : int
        Recommended number of clusters.
    results : dict
        Dictionary with keys 'k', 'gap', 'se' containing arrays of results.
        
    Examples
    --------
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> cluster_func = lambda X, k: AgglomerativeClustering(n_clusters=k).fit_predict(X)
    >>> optimal_k, results = find_optimal_k(X, cluster_func, k_range=range(2, 8))
    >>> print(f"Optimal k: {optimal_k}")
    """
    gs = SundarTibshiraniGapStatistic(
        pca_sampling=pca_sampling,
        use_user_labels=True,
        return_params=True,
        random_state=random_state
    )
    
    ks, gaps, ses = [], [], []
    
    for k in k_range:
        labels = clustering_func(X, k)
        gap, params = gs.compute_gap_statistic(X, labels, k=k, B=B)
        ks.append(k)
        gaps.append(gap)
        ses.append(params['sim_sks'])
    
    results = {'k': np.array(ks), 'gap': np.array(gaps), 'se': np.array(ses)}
    
    # Find optimal k based on criterion
    if criterion == 'gap':
        # Standard Gap criterion
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - ses[i + 1]:
                return ks[i], results
        # If no k satisfies criterion, return last k
        return ks[-1], results
        
    elif criterion == 'maxgap':
        optimal_idx = np.argmax(gaps)
        return ks[optimal_idx], results
        
    elif criterion == 'elbow':
        # Second derivative method
        if len(gaps) < 3:
            return ks[np.argmax(gaps)], results
        second_deriv = np.diff(gaps, 2)
        optimal_idx = np.argmax(second_deriv) + 1
        return ks[optimal_idx], results
        
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
