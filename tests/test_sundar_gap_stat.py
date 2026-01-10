"""
Test suite for Sundar-Tibshirani Gap Statistic
"""

import numpy as np
import pytest
from sundar_gap_stat import SundarTibshiraniGapStatistic, gap_statistic, find_optimal_k
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


class TestReproducibility:
    """Test that the same random_state produces identical results."""
    
    def test_same_random_state_same_result(self):
        """Two instances with the same seed must produce bit-for-bit identical results."""
        # Generate test data
        X, y = make_blobs(n_samples=100, n_features=4, centers=3, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create two instances with the same random_state
        gs1 = SundarTibshiraniGapStatistic(
            random_state=7142,
            return_params=True,
            use_user_labels=True
        )
        gs2 = SundarTibshiraniGapStatistic(
            random_state=7142,
            return_params=True,
            use_user_labels=True
        )
        
        # Compute gap statistics
        gap1, params1 = gs1.compute_gap_statistic(X_scaled, y, k=3, B=50)
        gap2, params2 = gs2.compute_gap_statistic(X_scaled, y, k=3, B=50)
        
        # Assert identical results
        assert gap1 == gap2, f"Gap values differ: {gap1} vs {gap2}"
        np.testing.assert_array_equal(
            params1['sim_Wks'], 
            params2['sim_Wks'],
            err_msg="Simulated Wk arrays are not identical"
        )
        assert params1['Wk'] == params2['Wk'], "Observed Wk values differ"
        assert params1['sd_k'] == params2['sd_k'], "Standard deviations differ"
        
    def test_different_random_state_different_result(self):
        """Different seeds should produce different results."""
        X, y = make_blobs(n_samples=100, n_features=4, centers=3, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        gs1 = SundarTibshiraniGapStatistic(random_state=7142, return_params=True)
        gs2 = SundarTibshiraniGapStatistic(random_state=9999, return_params=True)
        
        gap1, params1 = gs1.compute_gap_statistic(X_scaled, y, k=3, B=50)
        gap2, params2 = gs2.compute_gap_statistic(X_scaled, y, k=3, B=50)
        
        # Gap values should be different (with very high probability)
        assert not np.allclose(params1['sim_Wks'], params2['sim_Wks']), \
            "Different seeds should produce different simulated values"


class TestConvenienceFunction:
    """Test the gap_statistic convenience function."""
    
    def test_gap_statistic_function(self):
        """Test that convenience function returns gap and SE."""
        X, y = make_blobs(n_samples=100, n_features=4, centers=3, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        gap, se = gap_statistic(X_scaled, y, B=50, random_state=7142)
        
        assert isinstance(gap, float), "Gap should be a float"
        assert isinstance(se, float), "SE should be a float"
        assert se >= 0, "SE should be non-negative"
        
    def test_convenience_function_reproducibility(self):
        """Convenience function should also be reproducible."""
        X, y = make_blobs(n_samples=100, n_features=4, centers=3, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        gap1, se1 = gap_statistic(X_scaled, y, B=50, random_state=7142)
        gap2, se2 = gap_statistic(X_scaled, y, B=50, random_state=7142)
        
        assert gap1 == gap2, "Convenience function should be reproducible"
        assert se1 == se2, "SE should be reproducible"


class TestInputValidation:
    """Test input validation."""
    
    def test_invalid_B_type(self):
        """B must be an integer."""
        X = np.random.randn(50, 4)
        labels = np.array([0] * 25 + [1] * 25)
        gs = SundarTibshiraniGapStatistic()
        
        with pytest.raises(TypeError):
            gs.compute_gap_statistic(X, labels, B=50.5)
            
    def test_B_too_large(self):
        """B > 500 should raise ValueError."""
        X = np.random.randn(50, 4)
        labels = np.array([0] * 25 + [1] * 25)
        gs = SundarTibshiraniGapStatistic()
        
        with pytest.raises(ValueError):
            gs.compute_gap_statistic(X, labels, B=501)
            
    def test_invalid_distance_metric(self):
        """Invalid metric string should raise ValueError."""
        with pytest.raises(ValueError):
            SundarTibshiraniGapStatistic(distance_metric='invalid_metric')
            
    def test_invalid_distance_metric_type(self):
        """Non-string, non-callable metric should raise TypeError."""
        with pytest.raises(TypeError):
            SundarTibshiraniGapStatistic(distance_metric=123)
            
    def test_list_input_conversion(self):
        """X provided as list should be converted to array."""
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        labels = np.array([0, 0, 1, 1])
        gs = SundarTibshiraniGapStatistic()
        
        # Should not raise
        gap = gs.compute_gap_statistic(X, labels, B=20)
        assert isinstance(gap, float)


class TestSundarTibshiraniVsOriginal:
    """Test the difference between Sundar-Tibshirani and original."""
    
    def test_use_user_labels_true(self):
        """With use_user_labels=True, arbitrary labels can be evaluated."""
        X, _ = make_blobs(n_samples=100, n_features=4, centers=3, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use hierarchical clustering labels
        agg = AgglomerativeClustering(n_clusters=3)
        labels = agg.fit_predict(X_scaled)
        
        gs = SundarTibshiraniGapStatistic(use_user_labels=True)
        gap = gs.compute_gap_statistic(X_scaled, labels, k=3, B=30)
        
        assert isinstance(gap, float)
        
    def test_use_user_labels_false(self):
        """With use_user_labels=False, k-means is applied to reference data."""
        X, y = make_blobs(n_samples=100, n_features=4, centers=3, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        gs = SundarTibshiraniGapStatistic(use_user_labels=False, random_state=42)
        gap = gs.compute_gap_statistic(X_scaled, y, k=3, B=30)
        
        assert isinstance(gap, float)


class TestFindOptimalK:
    """Test the find_optimal_k function."""
    
    def test_find_optimal_k_returns_valid_k(self):
        """Should return k within the specified range."""
        X, _ = make_blobs(n_samples=150, n_features=4, centers=3, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        cluster_func = lambda X, k: KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        
        optimal_k, results = find_optimal_k(
            X_scaled, cluster_func, k_range=range(2, 6), B=30, random_state=7142
        )
        
        assert 2 <= optimal_k <= 5, f"Optimal k={optimal_k} outside range [2, 5]"
        assert 'gap' in results
        assert 'se' in results
        assert 'k' in results
        
    def test_find_optimal_k_maxgap_criterion(self):
        """Test maxgap criterion."""
        X, _ = make_blobs(n_samples=150, n_features=4, centers=3, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        cluster_func = lambda X, k: KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        
        optimal_k, results = find_optimal_k(
            X_scaled, cluster_func, k_range=range(2, 6), B=30, 
            criterion='maxgap', random_state=7142
        )
        
        # maxgap should return k with highest gap
        max_idx = np.argmax(results['gap'])
        assert optimal_k == results['k'][max_idx]


class TestDistanceMetrics:
    """Test different distance metrics."""
    
    def test_euclidean_metric(self):
        """Default euclidean metric should work."""
        X, y = make_blobs(n_samples=50, n_features=4, centers=2, random_state=42)
        gs = SundarTibshiraniGapStatistic(distance_metric='euclidean')
        gap = gs.compute_gap_statistic(X, y, B=20)
        assert isinstance(gap, float)
        
    def test_manhattan_metric(self):
        """Manhattan metric should work."""
        X, y = make_blobs(n_samples=50, n_features=4, centers=2, random_state=42)
        gs = SundarTibshiraniGapStatistic(distance_metric='manhattan')
        gap = gs.compute_gap_statistic(X, y, B=20)
        assert isinstance(gap, float)
        
    def test_cosine_metric(self):
        """Cosine metric should work."""
        X, y = make_blobs(n_samples=50, n_features=4, centers=2, random_state=42)
        gs = SundarTibshiraniGapStatistic(distance_metric='cosine')
        gap = gs.compute_gap_statistic(X, y, B=20)
        assert isinstance(gap, float)
        
    def test_custom_metric(self):
        """Custom callable metric should work."""
        from sklearn.metrics import pairwise_distances
        
        def custom_metric(X, Y):
            return pairwise_distances(X, Y, metric='chebyshev')
        
        X, y = make_blobs(n_samples=50, n_features=4, centers=2, random_state=42)
        gs = SundarTibshiraniGapStatistic(distance_metric=custom_metric)
        gap = gs.compute_gap_statistic(X, y, B=20)
        assert isinstance(gap, float)


class TestPCASampling:
    """Test PCA sampling option."""
    
    def test_pca_sampling_enabled(self):
        """PCA sampling should work when enabled."""
        X, y = make_blobs(n_samples=50, n_features=4, centers=2, random_state=42)
        gs = SundarTibshiraniGapStatistic(pca_sampling=True)
        gap = gs.compute_gap_statistic(X, y, B=20)
        assert isinstance(gap, float)
        
    def test_pca_sampling_disabled(self):
        """Should work without PCA sampling."""
        X, y = make_blobs(n_samples=50, n_features=4, centers=2, random_state=42)
        gs = SundarTibshiraniGapStatistic(pca_sampling=False)
        gap = gs.compute_gap_statistic(X, y, B=20)
        assert isinstance(gap, float)
        
    def test_standardize_within_pca(self):
        """Standardization within PCA should work."""
        X, y = make_blobs(n_samples=50, n_features=4, centers=2, random_state=42)
        gs = SundarTibshiraniGapStatistic(
            pca_sampling=True, 
            standardize_within_pca=True
        )
        gap = gs.compute_gap_statistic(X, y, B=20)
        assert isinstance(gap, float)


class TestRepr:
    """Test string representation."""
    
    def test_repr(self):
        """__repr__ should return informative string."""
        gs = SundarTibshiraniGapStatistic(
            distance_metric='manhattan',
            pca_sampling=False,
            use_user_labels=True
        )
        repr_str = repr(gs)
        assert 'SundarTibshiraniGapStatistic' in repr_str
        assert 'manhattan' in repr_str
        assert 'pca_sampling=False' in repr_str


if __name__ == '__main__':
    import sys
    # Handle both script execution and interactive sessions
    if hasattr(sys.modules[__name__], '__file__'):
        pytest.main([__file__, '-v'])
    else:
        pytest.main(['-v'])
