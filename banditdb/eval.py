"""
banditdb.eval — Offline Policy Evaluation utilities.

Reads directly from the Parquet files produced by POST /checkpoint.

Install with eval dependencies:
    pip install "banditdb-python[eval]"

Three estimators, increasing in statistical efficiency:

    replay(df)              Li et al. (2010) replay method. Unbiased, low coverage —
                            accepts each interaction with probability (1/K) / propensity.
                            Use as a sanity-check baseline.

    ips(df, clip)           Self-normalised Inverse Propensity Scoring (SNIPS). Uses every
                            interaction with importance weights. clip=10.0 bounds variance
                            at a small bias cost — recommended default.

    doubly_robust(df, clip) Doubly Robust estimator. Fits a linear reward model, then
                            applies an IPS correction on residuals. Most statistically
                            efficient; consistent if either the reward model or the
                            propensities are correctly specified.

All three functions:
  - Accept a Polars or pandas DataFrame loaded from a BanditDB Parquet export
  - Evaluate against a uniform random target policy (the unbiased baseline)
  - Raise ValueError for Thompson Sampling campaigns (propensity column is null)
  - Return an OPEResult with estimate, std_error, n_used, n_total, method

Quick start:
    import polars as pl
    from banditdb.eval import ips, doubly_robust

    df = pl.read_parquet("/data/exports/my_campaign.parquet")

    # How much reward would a uniform random policy have earned?
    baseline = ips(df)
    print(baseline)  # OPEResult(method='ips', estimate=0.41, std_error=0.02, ...)

    # Compare to your logging policy's raw observed reward:
    print("Observed:", df["reward"].mean())
    # If observed >> baseline.estimate, the campaign has learned something real.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class OPEResult:
    """Result of an offline policy evaluation run."""
    estimate:  float  # estimated mean reward of the target (uniform) policy
    std_error: float  # standard error of the estimate
    n_used:    int    # interactions used in the estimate
    n_total:   int    # total interactions in the dataset (after null-propensity filter)
    method:    str    # estimator name: "replay", "ips", or "doubly_robust"

    def __repr__(self) -> str:
        coverage = 100.0 * self.n_used / self.n_total if self.n_total > 0 else 0.0
        return (
            f"OPEResult(method='{self.method}', estimate={self.estimate:.4f}, "
            f"std_error={self.std_error:.4f}, "
            f"coverage={coverage:.1f}% [{self.n_used}/{self.n_total}])"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_numpy() -> None:
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise ImportError(
            "banditdb.eval requires numpy. "
            "Install with: pip install \"banditdb-python[eval]\""
        )


def _to_arrays(df) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (rewards, propensities, arm_ids, features) as numpy arrays from a
    Polars or pandas DataFrame. Returns raw arrays — null handling done by caller.
    """
    # Detect Polars vs pandas by duck-typing
    if hasattr(df, "to_pandas"):
        # Polars: nulls in Float64 columns come out as np.nan
        rewards      = df["reward"].to_numpy().astype(np.float64)
        propensities = df["propensity"].to_numpy(allow_copy=True).astype(np.float64)
        arm_ids      = df["arm_id"].to_numpy().astype(str)
        feat_cols    = sorted(c for c in df.columns if c.startswith("feature_"))
        features     = (
            np.column_stack([df[c].to_numpy() for c in feat_cols]).astype(np.float64)
            if feat_cols else np.empty((len(rewards), 0), dtype=np.float64)
        )
    else:
        # pandas: NaN for nulls
        rewards      = df["reward"].to_numpy(dtype=np.float64)
        propensities = df["propensity"].to_numpy(dtype=np.float64)
        arm_ids      = df["arm_id"].to_numpy().astype(str)
        feat_cols    = sorted(c for c in df.columns if c.startswith("feature_"))
        features     = (
            df[feat_cols].to_numpy(dtype=np.float64)
            if feat_cols else np.empty((len(rewards), 0), dtype=np.float64)
        )
    return rewards, propensities, arm_ids, features


def _validate_and_filter(
    df,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Load and validate a BanditDB Parquet DataFrame.

    Returns:
        rewards, propensities, arm_ids, features — filtered to rows with valid propensity
        n_total  — row count before filtering
        K        — number of distinct arms observed
    """
    rewards, propensities, arm_ids, features = _to_arrays(df)
    n_total = len(rewards)

    null_mask = np.isnan(propensities)
    if null_mask.all():
        raise ValueError(
            "All propensity values are null. This is a legacy Thompson Sampling export "
            "from before adaptive Monte Carlo propensity logging was added. "
            "Re-export from a current BanditDB instance to obtain propensity scores, "
            "or use causal_analysis.py which does not require logged propensities."
        )
    if null_mask.any():
        warnings.warn(
            f"{int(null_mask.sum())} of {n_total} rows have null propensity "
            "(legacy TS records from before adaptive Monte Carlo logging). "
            "These rows will be excluded from the IPS/SNIPS estimate.",
            stacklevel=3,
        )

    valid        = ~null_mask
    rewards      = rewards[valid]
    propensities = propensities[valid]
    arm_ids      = arm_ids[valid]
    features     = features[valid]
    K            = len(np.unique(arm_ids))
    return rewards, propensities, arm_ids, features, int(valid.sum()), K


def _snips(rewards: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """Self-normalised IPS point estimate and standard error (delta method)."""
    w_sum = weights.sum()
    if w_sum == 0.0:
        return 0.0, 0.0
    estimate  = float((weights * rewards).sum() / w_sum)
    n         = len(rewards)
    residuals = weights * (rewards - estimate) / w_sum
    std_error = float(np.sqrt((residuals ** 2).sum()) * np.sqrt(n / max(n - 1, 1)))
    return estimate, std_error


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def replay(df, num_arms: Optional[int] = None, seed: int = 42) -> OPEResult:
    """
    Li et al. (2010) replay method against a uniform random target policy.

    Each logged interaction is accepted with probability min(1, (1/K) / propensity).
    Accepted interactions form an unbiased sample from the uniform target policy.
    Low coverage is expected and normal — use ips() if you need more precision.

    Parameters
    ----------
    df : Polars or pandas DataFrame from a BanditDB Parquet export.
    num_arms : Number of arms K. Inferred from unique arm_id values if omitted.
    seed : Random seed for reproducible acceptance sampling. Default 42.

    Returns
    -------
    OPEResult — estimated mean reward of a uniform random policy.
    """
    _require_numpy()
    rewards, propensities, arm_ids, _, n_total, k_inferred = _validate_and_filter(df)
    K = num_arms or k_inferred

    acceptance = np.minimum(1.0, (1.0 / K) / propensities)
    rng        = np.random.default_rng(seed=seed)
    accepted   = rng.random(n_total) < acceptance
    n_used     = int(accepted.sum())

    if n_used == 0:
        raise ValueError(
            "No interactions accepted by the replay filter. "
            "Propensity values may be miscalibrated."
        )
    if n_used < 30:
        warnings.warn(
            f"Only {n_used} interactions accepted ({100.0 * n_used / n_total:.1f}% coverage). "
            "Estimate may be unreliable — collect more data or use ips().",
            stacklevel=2,
        )

    used      = rewards[accepted]
    estimate  = float(used.mean())
    std_error = float(used.std() / np.sqrt(n_used)) if n_used > 1 else 0.0

    return OPEResult(
        estimate=estimate, std_error=std_error,
        n_used=n_used, n_total=n_total, method="replay",
    )


def ips(
    df,
    clip: Optional[float] = 10.0,
    num_arms: Optional[int] = None,
) -> OPEResult:
    """
    Self-normalised Inverse Propensity Scoring (SNIPS) against a uniform target policy.

    Uses every interaction with an importance weight (1/K) / propensity rather than
    discarding interactions like replay(). Propensity clipping (default clip=10.0)
    reduces variance at the cost of a small bias — set clip=None to disable.

    Parameters
    ----------
    df : Polars or pandas DataFrame from a BanditDB Parquet export.
    clip : Maximum importance weight. Recommended 5–20. None disables clipping.
    num_arms : Number of arms K. Inferred from unique arm_id values if omitted.

    Returns
    -------
    OPEResult — estimated mean reward of a uniform random policy.
    """
    _require_numpy()
    rewards, propensities, _, __, n_total, k_inferred = _validate_and_filter(df)
    K = num_arms or k_inferred

    weights = (1.0 / K) / propensities
    if clip is not None:
        weights = np.minimum(weights, float(clip))

    estimate, std_error = _snips(rewards, weights)
    return OPEResult(
        estimate=estimate, std_error=std_error,
        n_used=n_total, n_total=n_total, method="ips",
    )


def doubly_robust(
    df,
    clip: Optional[float] = 10.0,
    num_arms: Optional[int] = None,
) -> OPEResult:
    """
    Doubly Robust (DR) estimator against a uniform random target policy.

    Fits a linear reward model μ̂(context, arm) via least squares, then applies an
    IPS correction on the residuals. Consistent if either the reward model or the
    propensities are correctly specified — more statistically efficient than ips() alone.

    The direct model uses the feature columns (feature_0 … feature_N) plus a one-hot
    arm encoding and a bias term. A linear model is intentionally simple: it must not
    overfit, or the DR correction will amplify rather than reduce bias.

    Parameters
    ----------
    df : Polars or pandas DataFrame from a BanditDB Parquet export.
    clip : Maximum importance weight for the IPS correction term. Default 10.0.
    num_arms : Number of arms K. Inferred from unique arm_id values if omitted.

    Returns
    -------
    OPEResult — estimated mean reward of a uniform random policy.
    """
    _require_numpy()
    rewards, propensities, arm_ids, features, n_total, k_inferred = _validate_and_filter(df)
    K = num_arms or k_inferred
    n = n_total  # already filtered

    # One-hot encode arms
    unique_arms = sorted(set(arm_ids))
    arm_to_idx  = {a: i for i, a in enumerate(unique_arms)}
    arm_ohe     = np.zeros((n, len(unique_arms)), dtype=np.float64)
    for i, a in enumerate(arm_ids):
        arm_ohe[i, arm_to_idx[a]] = 1.0

    # Design matrix: [context features | arm one-hot | bias]
    bias   = np.ones((n, 1), dtype=np.float64)
    design = np.hstack([features, arm_ohe, bias]) if features.shape[1] > 0 else np.hstack([arm_ohe, bias])

    # Fit linear reward model
    try:
        beta, _, _, _ = np.linalg.lstsq(design, rewards, rcond=None)
        mu_hat = design @ beta
    except np.linalg.LinAlgError:
        warnings.warn(
            "Reward model fitting failed (singular design matrix). Falling back to ips().",
            stacklevel=2,
        )
        return ips(df, clip=clip, num_arms=num_arms)

    # Direct Model (DM): predict reward for each arm under the uniform target policy
    dm_per_arm = []
    for arm_name in unique_arms:
        ohe      = np.zeros(len(unique_arms), dtype=np.float64)
        ohe[arm_to_idx[arm_name]] = 1.0
        ohe_mat  = np.tile(ohe, (n, 1))
        des_arm  = np.hstack([features, ohe_mat, bias]) if features.shape[1] > 0 else np.hstack([ohe_mat, bias])
        dm_per_arm.append(des_arm @ beta)

    # Average over arms (uniform target policy weights each arm equally)
    dm_mean = np.mean(dm_per_arm, axis=0)

    # IPS correction on residuals
    weights = (1.0 / K) / propensities
    if clip is not None:
        weights = np.minimum(weights, float(clip))

    dr_values  = dm_mean + weights * (rewards - mu_hat)
    estimate   = float(dr_values.mean())
    std_error  = float(dr_values.std() / np.sqrt(max(n - 1, 1)))

    return OPEResult(
        estimate=estimate, std_error=std_error,
        n_used=n, n_total=n, method="doubly_robust",
    )
