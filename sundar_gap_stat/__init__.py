"""
Sundar-Tibshirani Gap Statistic
===============================

A generalized Gap Statistic for evaluating any cluster solution.

Main Classes
------------
SundarTibshiraniGapStatistic
    The main class for computing Gap Statistics.

Functions
---------
gap_statistic
    Convenience function for quick Gap computation.
find_optimal_k
    Find optimal number of clusters using Gap criterion.

Example
-------
>>> from sundar_gap_stat import SundarTibshiraniGapStatistic, gap_statistic
>>> from sklearn.cluster import AgglomerativeClustering
>>> 
>>> # Any clustering algorithm
>>> labels = AgglomerativeClustering(n_clusters=3).fit_predict(X)
>>> 
>>> # Compute Gap Statistic
>>> gap, se = gap_statistic(X, labels, B=100)
>>> print(f"Gap: {gap:.3f} (SE: {se:.3f})")

References
----------
Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number 
of clusters in a data set via the gap statistic. Journal of the Royal 
Statistical Society: Series B, 63(2), 411-423.
"""

from .sundar_gap_stat import (
    SundarTibshiraniGapStatistic,
    gap_statistic,
    find_optimal_k,
    __version__,
    __author__
)

__all__ = [
    'SundarTibshiraniGapStatistic',
    'gap_statistic',
    'find_optimal_k',
    '__version__',
    '__author__'
]
