"""
Pure statistical helper functions for the Track Record dashboard.

No Streamlit, no pandas, no network I/O — importable anywhere including tests.
"""
from __future__ import annotations

import math


def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval.

    Formula (confirmed 2026-06-22):
        denom  = 1 + z²/n
        center = (p̂ + z²/2n) / denom
        margin = (z/denom) · √(p̂(1−p̂)/n + z²/4n²)
        CI     = [center − margin, center + margin]  clamped to [0,1]

    Returns (0.0, 0.0) for n == 0.
    """
    if n == 0:
        return 0.0, 0.0
    p      = hits / n
    denom  = 1.0 + z ** 2 / n
    center = (p + z ** 2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z ** 2 / (4.0 * n ** 2))
    return max(0.0, center - margin), min(1.0, center + margin)
