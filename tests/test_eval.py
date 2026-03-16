"""
Tests for banditdb.eval — Offline Policy Evaluation utilities.

Uses synthetic Polars DataFrames that mimic the schema written by
POST /checkpoint: interaction_id | arm_id | reward | predicted_at |
rewarded_at | propensity | feature_0 … feature_N
"""

import math
import warnings

import numpy as np
import polars as pl
import pytest

from banditdb.eval import OPEResult, doubly_robust, ips, replay


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 500,
    num_arms: int = 3,
    feature_dim: int = 2,
    seed: int = 0,
    include_null_propensity: bool = False,
) -> pl.DataFrame:
    """
    Synthetic BanditDB Parquet schema. Propensities are uniform (1/K) so that
    IPS weights are exactly 1.0 — the IPS estimate should match the raw mean.
    """
    rng     = np.random.default_rng(seed)
    arms    = [f"arm_{i}" for i in range(num_arms)]
    arm_ids = [arms[i % num_arms] for i in range(n)]
    rewards = rng.uniform(0.0, 1.0, n).tolist()
    # Uniform logging policy: propensity = 1/K for every row
    propensities: list = [1.0 / num_arms] * n

    if include_null_propensity:
        # Simulate a few TS rows mixed in
        for i in range(0, n, 10):
            propensities[i] = None  # type: ignore[call-overload]

    data: dict = {
        "interaction_id": [f"iid-{i}" for i in range(n)],
        "arm_id":         arm_ids,
        "reward":         rewards,
        "predicted_at":   [1_700_000_000 + i for i in range(n)],
        "rewarded_at":    [1_700_000_000 + i + 60 for i in range(n)],
        "propensity":     propensities,
    }
    for f in range(feature_dim):
        data[f"feature_{f}"] = rng.uniform(0.0, 1.0, n).tolist()

    schema = {"propensity": pl.Float64}
    return pl.DataFrame(data, schema_overrides=schema)


def _all_null_df(n: int = 100, num_arms: int = 4) -> pl.DataFrame:
    """DataFrame where every propensity is null (TS campaign)."""
    return pl.DataFrame({
        "interaction_id": [f"iid-{i}" for i in range(n)],
        "arm_id":         [f"arm_{i % num_arms}" for i in range(n)],
        "reward":         np.random.default_rng(1).uniform(0, 1, n).tolist(),
        "predicted_at":   list(range(n)),
        "rewarded_at":    list(range(n)),
        "propensity":     [None] * n,
    }, schema_overrides={"propensity": pl.Float64})


# ---------------------------------------------------------------------------
# OPEResult
# ---------------------------------------------------------------------------

class TestOPEResult:
    def test_repr_shows_coverage(self):
        r = OPEResult(estimate=0.5, std_error=0.02, n_used=80, n_total=100, method="ips")
        assert "80.0%" in repr(r)
        assert "ips" in repr(r)

    def test_repr_zero_total(self):
        r = OPEResult(estimate=0.0, std_error=0.0, n_used=0, n_total=0, method="replay")
        assert "0.0%" in repr(r)


# ---------------------------------------------------------------------------
# replay()
# ---------------------------------------------------------------------------

class TestReplay:
    def test_returns_ope_result(self):
        df = _make_df()
        result = replay(df)
        assert isinstance(result, OPEResult)
        assert result.method == "replay"

    def test_estimate_in_unit_interval(self):
        df = _make_df()
        result = replay(df)
        assert 0.0 <= result.estimate <= 1.0

    def test_std_error_non_negative(self):
        df = _make_df()
        assert replay(df).std_error >= 0.0

    def test_n_total_matches_df_length(self):
        df = _make_df(n=400)
        result = replay(df)
        assert result.n_total == 400

    def test_n_used_leq_n_total(self):
        df = _make_df()
        result = replay(df)
        assert result.n_used <= result.n_total

    def test_ts_campaign_raises(self):
        df = _all_null_df()
        with pytest.raises(ValueError, match="Thompson Sampling"):
            replay(df)

    def test_mixed_null_warns(self):
        df = _make_df(include_null_propensity=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            replay(df)
        messages = " ".join(str(warning.message) for warning in w)
        assert "null propensity" in messages

    def test_reproducible_with_seed(self):
        df = _make_df(n=1000)
        r1 = replay(df, seed=99)
        r2 = replay(df, seed=99)
        assert r1.estimate == r2.estimate
        assert r1.n_used == r2.n_used

    def test_different_seeds_differ_with_partial_acceptance(self):
        # With propensities > 1/K, acceptance < 1.0, so seeds produce different samples.
        n = 400
        rng = np.random.default_rng(5)
        df = pl.DataFrame({
            "interaction_id": [f"i{i}" for i in range(n)],
            "arm_id":         ["arm_a", "arm_b"] * (n // 2),
            "reward":         rng.uniform(0, 1, n).tolist(),
            "predicted_at":   list(range(n)),
            "rewarded_at":    list(range(n)),
            # Propensity = 0.9 >> 1/K=0.5, so acceptance = (0.5/0.9) ≈ 0.56
            "propensity":     [0.9] * n,
        }, schema_overrides={"propensity": pl.Float64})
        r1 = replay(df, seed=0)
        r2 = replay(df, seed=1)
        assert r1.n_used != r2.n_used or r1.estimate != r2.estimate

    def test_uniform_propensity_acceptance_rate(self):
        # Under uniform logging (p=1/K), acceptance = (1/K)/(1/K) = 1.0 — all rows accepted
        df = _make_df(n=1000, num_arms=4)
        result = replay(df)
        assert result.n_used == result.n_total


# ---------------------------------------------------------------------------
# ips()
# ---------------------------------------------------------------------------

class TestIPS:
    def test_returns_ope_result(self):
        df = _make_df()
        assert isinstance(ips(df), OPEResult)
        assert ips(df).method == "ips"

    def test_uses_all_rows(self):
        df = _make_df(n=300)
        result = ips(df)
        assert result.n_used == result.n_total == 300

    def test_uniform_propensity_matches_raw_mean(self):
        # When logging propensity == 1/K and target is uniform, IPS weights = 1.
        # SNIPS estimate should equal the raw reward mean.
        df    = _make_df(n=1000, num_arms=4, seed=7)
        raw   = df["reward"].mean()
        result = ips(df, clip=None)
        assert abs(result.estimate - raw) < 1e-6

    def test_clipping_changes_estimate_for_skewed_propensities(self):
        # Build a df with very unequal propensities to make clipping matter
        n = 200
        rng = np.random.default_rng(42)
        propensities = rng.uniform(0.01, 0.99, n).tolist()
        df = pl.DataFrame({
            "interaction_id": [f"i{i}" for i in range(n)],
            "arm_id":         ["arm_a" if p > 0.5 else "arm_b" for p in propensities],
            "reward":         rng.uniform(0, 1, n).tolist(),
            "predicted_at":   list(range(n)),
            "rewarded_at":    list(range(n)),
            "propensity":     propensities,
        }, schema_overrides={"propensity": pl.Float64})
        r_clipped   = ips(df, clip=5.0)
        r_unclipped = ips(df, clip=None)
        # They won't be identical when clipping is active
        assert r_clipped.estimate != r_unclipped.estimate

    def test_ts_campaign_raises(self):
        with pytest.raises(ValueError, match="Thompson Sampling"):
            ips(_all_null_df())

    def test_estimate_finite(self):
        result = ips(_make_df())
        assert math.isfinite(result.estimate)
        assert math.isfinite(result.std_error)

    def test_num_arms_override_accepted(self):
        # SNIPS cancels K entirely (sum(c·w·r)/sum(c·w) = sum(w·r)/sum(w)),
        # so num_arms does not change the point estimate — but it must be accepted
        # without error and produce a valid result.
        df = _make_df(num_arms=3)
        r = ips(df, clip=None, num_arms=6)
        assert isinstance(r, OPEResult)
        assert math.isfinite(r.estimate)


# ---------------------------------------------------------------------------
# doubly_robust()
# ---------------------------------------------------------------------------

class TestDoublyRobust:
    def test_returns_ope_result(self):
        df = _make_df()
        assert isinstance(doubly_robust(df), OPEResult)
        assert doubly_robust(df).method == "doubly_robust"

    def test_uses_all_valid_rows(self):
        df = _make_df(n=300)
        assert doubly_robust(df).n_used == 300

    def test_estimate_finite(self):
        result = doubly_robust(_make_df())
        assert math.isfinite(result.estimate)
        assert math.isfinite(result.std_error)

    def test_estimate_in_plausible_range(self):
        # With uniform [0,1] rewards and a simple linear model, DR should be in [0,1]
        result = doubly_robust(_make_df(n=500, feature_dim=3))
        assert -0.5 <= result.estimate <= 1.5  # generous tolerance for small-sample variance

    def test_ts_campaign_raises(self):
        with pytest.raises(ValueError, match="Thompson Sampling"):
            doubly_robust(_all_null_df())

    def test_no_feature_columns(self):
        # A campaign with feature_dim=0 (only arm one-hot + bias in design matrix)
        df = _make_df(n=300, feature_dim=0)
        result = doubly_robust(df)
        assert math.isfinite(result.estimate)

    def test_mixed_null_warns_and_excludes(self):
        df = _make_df(n=500, include_null_propensity=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = doubly_robust(df)
        assert result.n_used < 500
        assert any("null propensity" in str(warning.message) for warning in w)

    def test_std_error_smaller_than_ips_for_linear_rewards(self):
        # For linearly predictable rewards, DR should be more efficient than IPS
        n   = 1000
        rng = np.random.default_rng(0)
        features = rng.uniform(0, 1, n)
        rewards  = 0.8 * features + rng.normal(0, 0.05, n)  # nearly linear
        rewards  = np.clip(rewards, 0, 1)
        arms     = ["arm_a" if f > 0.5 else "arm_b" for f in features]
        df = pl.DataFrame({
            "interaction_id": [f"i{i}" for i in range(n)],
            "arm_id":         arms,
            "reward":         rewards.tolist(),
            "predicted_at":   list(range(n)),
            "rewarded_at":    list(range(n)),
            "propensity":     [0.5] * n,
            "feature_0":      features.tolist(),
        }, schema_overrides={"propensity": pl.Float64})
        r_ips = ips(df, clip=None)
        r_dr  = doubly_robust(df, clip=None)
        assert r_dr.std_error <= r_ips.std_error * 1.1  # DR should be at least as efficient
