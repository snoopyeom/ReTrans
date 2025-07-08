import numpy as np
from typing import Sequence


def dtw_distance(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Dynamic Time Warping distance between two sequences."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def normality_score(samples: np.ndarray) -> float:
    """Return a simple normality score based on skew and kurtosis."""
    if samples.ndim > 1:
        samples = samples.reshape(-1)
    mean = np.mean(samples)
    std = np.std(samples) + 1e-8
    standardized = (samples - mean) / std
    skew = np.mean(standardized ** 3)
    kurt = np.mean(standardized ** 4) - 3
    return float(abs(skew) + abs(kurt))
