"""Paired scoring and multiple-comparison helpers.

The paper judges a decision by downstream performance on paired folds:

    d(r, k) = a(r, k) - b(r, k)

with ``a`` the action under test and ``b`` the no-processing control, and
reports the Wilcoxon signed-rank test with a paired t confidence interval.
Holm correction is applied within pre-specified families.  This module
implements those helpers; the decoder and the fold partition are supplied by
the reader.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats


def paired_differences(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Element-wise ``a - b`` after checking equal length."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired sequences must have the same length")
    return a - b


def wilcoxon_p(diffs: Sequence[float], method: str = "exact") -> float:
    """Two-sided Wilcoxon signed-rank p; zero differences are dropped.

    The paper reports the exact test.  ``method="exact"`` is the default;
    when the exact test is unavailable (ties with small samples), the call
    falls back to ``"auto"`` and emits a warning rather than silently
    changing the test.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 1.0
    try:
        return float(stats.wilcoxon(d, method=method).pvalue)
    except ValueError as exc:
        warnings.warn(f"exact Wilcoxon unavailable ({exc}); falling back to auto")
        return float(stats.wilcoxon(d, method="auto").pvalue)


def paired_t_ci(diffs: Sequence[float], alpha: float) -> Tuple[float, float]:
    """Two-sided paired t confidence interval at level ``1 - alpha``."""
    d = np.asarray(diffs, dtype=float)
    n = d.size
    if n < 2:
        return (float("nan"), float("nan"))
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return (float(mean - tcrit * se), float(mean + tcrit * se))


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjustment, returned in the input order."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def family_test(
    family: Dict[str, Sequence[float]],
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    """Run one family of paired tests and Holm-correct within it.

    ``family`` maps a name to its paired differences.  Returns, per name,
    ``mean``, ``p``, ``p_holm``, ``ci_low``, ``ci_high``.

    ``alpha`` is the significance level; 0.05 is conventional (Holm, 1979).
    """
    names = list(family)
    raw = [wilcoxon_p(family[n]) for n in names]
    adjusted = holm(raw)
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        diffs = np.asarray(family[name], dtype=float)
        lo, hi = paired_t_ci(diffs, alpha)
        out[name] = {
            "mean": float(diffs.mean()) if diffs.size else float("nan"),
            "p": float(raw[i]),
            "p_holm": float(adjusted[i]),
            "ci_low": lo,
            "ci_high": hi,
        }
    return out


def families_test(
    families: Dict[str, Dict[str, Sequence[float]]],
    alpha: float,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run several pre-specified families, Holm-correcting each one.

    The paper corrects the primary family (baseline, random) and the
    secondary family (the other controls) separately; pass one entry per
    family, e.g. ``{"primary": {...}, "secondary": {...}}``.
    """
    return {name: family_test(family, alpha) for name, family in families.items()}
