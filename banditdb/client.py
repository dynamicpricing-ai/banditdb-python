"""
banditdb.client — Synchronous Python client for BanditDB.

Quickstart:
    from banditdb import Client, NeuralLinUCBConfig, NeuralTSConfig, ProgressiveConfig

    db = Client("http://localhost:8080", api_key="...")

    # Simple LinUCB campaign
    db.create_campaign("prices", ["10", "15", "20"], feature_dim=5)

    arm, iid = db.predict("prices", [0.3, 0.7, 0.1, 0.9, 0.4])
    db.reward(iid, 1.0)

    # Neural contextual bandit
    cfg = NeuralLinUCBConfig(context_dim=5, embed_dim=32)
    db.create_campaign("prices_neural", ["10", "15", "20"], feature_dim=5, algorithm=cfg)

    # NeuralTS: same config, posterior-sampling exploration
    cfg = NeuralTSConfig(context_dim=5, embed_dim=32)
    db.create_campaign("prices_nts", ["10", "15", "20"], feature_dim=5, algorithm=cfg)

    # Self-tuning tournament (LinUCB vs NeuralLinUCB)
    cfg = ProgressiveConfig(
        base="linucb",
        challenger=NeuralLinUCBConfig(context_dim=5, embed_dim=32),
    )
    db.create_campaign("prices_prog", ["10", "15", "20"], feature_dim=5, algorithm=cfg)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import APIError, ConnectionError, TimeoutError

# Mirrors BANDITDB_MAX_CONTEXT_MAGNITUDE on the server (default 1e6). Checking here
# turns a round trip and a 400 into an immediate, local error naming the offending
# index. Finiteness alone is not enough: a value near 1e155 is finite but squares to
# infinity inside the server's rank-one update, and the resulting NaN persists
# through the checkpoint and survives restart.
MAX_CONTEXT_MAGNITUDE = 1e6


def normalize_context(context: List[float]) -> List[float]:
    """
    Scale a context vector to unit L2 norm.

    Worth doing. LinUCB scores an arm as ``θ·x + α·√(xᵀA⁻¹x)``; the exploration term
    scales with ``‖x‖`` and the regret analysis assumes ``‖x‖ ≤ 1``. Feeding
    unnormalised or heavy-tailed features lets rare high-norm rows dominate arm
    selection, and the model still converges — just far more slowly, with no error to
    tell you. On the UCI shuttle benchmark this single change moved cumulative regret
    from 2,026 to 709 with no other difference.

    If your features have very different ranges, min-max scale each into [0, 1]
    first, then call this. An all-zero vector is returned unchanged.

    Example::

        db.predict("c", normalize_context([age, income, clicks]))
    """
    norm = math.sqrt(sum(v * v for v in context))
    if norm == 0.0 or not math.isfinite(norm):
        return list(context)
    return [v / norm for v in context]


def _validate_context(context: List[float]) -> None:
    """Reject contexts the server would reject, with a clearer message."""
    if not context:
        raise ValueError("context must not be empty")
    for i, v in enumerate(context):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"context[{i}] must be a number, got {type(v).__name__}")
        if not math.isfinite(v):
            raise ValueError(
                f"context[{i}] is {v}. NaN and infinity propagate into the arm's "
                "covariance matrix and cannot be cleared without deleting the campaign."
            )
        if abs(v) > MAX_CONTEXT_MAGNITUDE:
            raise ValueError(
                f"context[{i}] magnitude {abs(v)} exceeds {MAX_CONTEXT_MAGNITUDE:g}. "
                "Squared terms would overflow during the server's rank-one update. "
                "Scale your features — normalize_context() is usually what you want."
            )


def _validate_reward(reward: float) -> None:
    if not isinstance(reward, (int, float)) or isinstance(reward, bool):
        raise ValueError(f"reward must be a number, got {type(reward).__name__}")
    if not math.isfinite(reward):
        raise ValueError(f"reward must be finite, got {reward}")
    if not 0.0 <= reward <= 1.0:
        raise ValueError(
            f"reward {reward} is outside [0.0, 1.0]. Server 2.0.0+ rejects this with "
            "400 — the confidence bounds and the SNIPS estimator both assume the range. "
            "Rescale your outcome, e.g. revenue / max_revenue."
        )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Algorithm config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NeuralLinUCBConfig:
    """
    Configuration for the NeuralLinUCB contextual bandit algorithm.

    NeuralLinUCB learns a deep embedding of the context vector (Algorithm 1),
    then applies LinUCB in the embedding space. The MLP is retrained every
    `retrain_every` rewards using Algorithm 2 (gradient descent + Sherman-Morrison
    warm start). Use this when the reward function is non-linear in the raw features.

    Parameters
    ----------
    context_dim   : Number of floats in the context vector you pass to predict().
    embed_dim     : Size of the learned embedding (arm matrix dimension). Default 32.
    hidden_dim    : Width of each hidden layer in the MLP. Default 128.
    hidden_layers : Number of hidden layers. Default 2.
    retrain_every : Retrain the MLP after this many cumulative reward updates. Default 200.
    retrain_steps : Gradient descent steps per retrain. Default 100.
    learning_rate : AdamW learning rate. Default 1e-3.
    lambda_reg    : L2 regularisation strength anchoring weights near initialisation. Default 1.0.
    """
    context_dim:   int
    embed_dim:     int   = 32
    hidden_dim:    int   = 128
    hidden_layers: int   = 2
    retrain_every: int   = 200
    retrain_steps: int   = 100
    learning_rate: float = 1e-3
    lambda_reg:    float = 1.0

    _API_KEY = "neural_lin_ucb"

    def _to_api(self) -> dict:
        return {
            self._API_KEY: {
                "context_dim":   self.context_dim,
                "embed_dim":     self.embed_dim,
                "hidden_dim":    self.hidden_dim,
                "hidden_layers": self.hidden_layers,
                "retrain_every": self.retrain_every,
                "retrain_steps": self.retrain_steps,
                "learning_rate": self.learning_rate,
                "lambda":        self.lambda_reg,
            }
        }


@dataclass
class NeuralTSConfig(NeuralLinUCBConfig):
    """
    Configuration for NeuralTS, Neural Thompson Sampling (Zhang et al., ICLR 2021).

    Same MLP embedding and retrain procedure as NeuralLinUCB — the parameters are
    identical — but each arm is scored by a posterior draw ``w ~ N(θ, α²A⁻¹)`` in
    the embedding space instead of an upper confidence bound. ``alpha`` on
    create_campaign() scales the spread of that draw.

    Stochastic exploration costs more regret early on and converges further over
    long runs (20K+ interactions). Prefer NeuralLinUCBConfig when early mistakes are
    expensive; prefer this when long-run performance matters more. Like the other
    neural algorithms it needs a server built with the ``neural`` feature — check
    ``server_info()["features"]``.

    Parameters: see NeuralLinUCBConfig.

    Example::

        cfg = NeuralTSConfig(context_dim=128, embed_dim=16)
        db.create_campaign("churn", arms, feature_dim=128, algorithm=cfg)
    """
    _API_KEY = "neural_thompson_sampling"


@dataclass
class ProgressiveConfig:
    """
    Configuration for the Progressive self-tuning tournament algorithm.

    Runs a base model and a challenger in parallel ("shadow learning"). Every
    reward updates both models. At each checkpoint the server evaluates both
    with SNIPS (Self-Normalised IPS). If the challenger wins `required_wins`
    consecutive rounds by more than 10%, one `step_bps` of traffic shifts to
    the challenger — and vice-versa for the base. Traffic never drops below 10%
    (exploration floor) or exceeds 90% (promotion ceiling).

    Parameters
    ----------
    base          : Base algorithm. "linucb", "thompson_sampling", NeuralLinUCBConfig,
                    or NeuralTSConfig.
    challenger    : Challenger algorithm. Same options.
    min_obs       : Minimum buffer entries per arm before any traffic shift fires. Default 100.
    required_wins : Consecutive checkpoint wins required to earn one traffic step. Default 3.
    step_bps      : Traffic delta per confirmed win run, in basis points (1000 = 10%). Default 1000.
    """
    base:          Union[Literal["linucb", "thompson_sampling"], NeuralLinUCBConfig] = "linucb"
    challenger:    Union[Literal["linucb", "thompson_sampling"], NeuralLinUCBConfig] = field(
        default_factory=lambda: NeuralLinUCBConfig(context_dim=1)  # placeholder; context_dim required
    )
    min_obs:       int = 100
    required_wins: int = 3
    step_bps:      int = 1000

    def _to_api(self) -> dict:
        def _algo(a):
            if isinstance(a, str):
                return a
            return a._to_api()
        return {
            "progressive": {
                "base":          _algo(self.base),
                "challenger":    _algo(self.challenger),
                "min_obs":       self.min_obs,
                "required_wins": self.required_wins,
                "step_bps":      self.step_bps,
            }
        }


Algorithm = Union[
    Literal["linucb", "thompson_sampling"],
    NeuralLinUCBConfig,
    NeuralTSConfig,
    ProgressiveConfig,
]

ArmStatus = Literal["active", "paused", "retired"]
ARM_STATUSES: Tuple[str, ...] = ("active", "paused", "retired")

def _serialise_algorithm(algo: Algorithm) -> Any:
    if isinstance(algo, str):
        return algo
    return algo._to_api()

# ---------------------------------------------------------------------------
# Budget constraints (Lagrangian pacing, server 2.3.0+)
# ---------------------------------------------------------------------------

def _check_number(value: Any, what: str, allow_zero: bool) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{what} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{what} must be finite and {bound}, got {value}")


def _validate_costs(costs: Dict[str, float], what: str) -> None:
    for key, cost in costs.items():
        _check_number(cost, f"{what}['{key}']", allow_zero=True)


@dataclass
class ResourceConstraint:
    """
    One budgeted resource in a pacing campaign: coupons, call-centre minutes,
    ad spend — anything an arm consumes when it is chosen.

    The server learns a shadow price per unit of this resource and subtracts
    ``price × cost`` from each arm's score, so costly arms are chosen only where
    their uplift is worth what they spend. An arm whose cost exceeds the remaining
    budget is not selectable.

    Parameters
    ----------
    name           : Resource name, unique within the campaign (e.g. "promo_budget").
    budget         : Total amount available over the horizon. Must be > 0.
    arm_costs      : Units consumed each time an arm is **predicted**, keyed by arm id.
                     Arms left out cost 0. Every key must be one of the campaign's arms.
    horizon        : Number of decisions the budget should last. Default 10,000.
    step_size      : Shadow-price learning rate. Default: auto-tuned from the horizon
                     (2 / √horizon).
    lambda_max     : Upper bound on the shadow price. Default: auto-tuned.
    initial_lambda : Starting shadow price. Default 0.0 (no penalty until spend
                     outruns the target rate).
    """
    name:           str
    budget:         float
    arm_costs:      Dict[str, float] = field(default_factory=dict)
    horizon:        int              = 10_000
    step_size:      Optional[float]  = None
    lambda_max:     Optional[float]  = None
    initial_lambda: Optional[float]  = None

    def _validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("resource name must be a non-empty string")
        what = f"resource '{self.name}'"
        _check_number(self.budget, f"{what} budget", allow_zero=False)
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon <= 0:
            raise ValueError(f"{what} horizon must be a positive integer, got {self.horizon!r}")
        if self.step_size is not None:
            _check_number(self.step_size, f"{what} step_size", allow_zero=False)
        if self.lambda_max is not None:
            _check_number(self.lambda_max, f"{what} lambda_max", allow_zero=False)
        if self.initial_lambda is not None:
            _check_number(self.initial_lambda, f"{what} initial_lambda", allow_zero=True)
        _validate_costs(self.arm_costs, f"{what} arm_costs")

    def _to_api(self) -> dict:
        out: dict = {
            "name":      self.name,
            "budget":    self.budget,
            "horizon":   self.horizon,
            "arm_costs": dict(self.arm_costs),
        }
        for key in ("step_size", "lambda_max", "initial_lambda"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


@dataclass
class PacingConfig:
    """
    Budget constraints for a campaign. Requires server 2.3.0+.

    Parameters
    ----------
    resources : One ResourceConstraint per budgeted resource.
    adaptive  : Target the remaining budget over the remaining decisions (True,
                default) rather than a fixed budget / horizon rate. Adaptive pacing
                recovers from early over- or under-spend.

    Behaviours worth knowing before you rely on it:

    - **Budget is spent at predict, not reward.** A recommendation you never act
      on still consumes budget.
    - **Budget never resets, and the config is immutable.** Once spent, costly arms
      stay unselectable until the campaign is recreated.
    - **Downgrading the server below 2.3.0 silently drops pacing** — older binaries
      ignore the field and the campaign becomes unconstrained without error.
    - Predicts on a pacing campaign return 503 when the server's WAL cannot take
      the extra consumption record; unconstrained campaigns are unaffected.

    Example::

        PacingConfig(resources=[
            ResourceConstraint(
                name="promo_budget",
                budget=5_000.0,
                horizon=100_000,
                arm_costs={"discount_10": 1.0, "discount_20": 2.0},
            ),
        ])
    """
    resources: List[ResourceConstraint]
    adaptive:  bool = True

    def _validate(self) -> None:
        seen = set()
        for r in self.resources:
            r._validate()
            if r.name in seen:
                raise ValueError(f"duplicate resource name: '{r.name}'")
            seen.add(r.name)

    def _to_api(self) -> dict:
        return {
            "resources": [r._to_api() for r in self.resources],
            "adaptive":  self.adaptive,
        }

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class Client:
    """
    Production-ready synchronous client for BanditDB.

    All methods raise:
        TimeoutError   — server did not respond within `timeout` seconds.
        ConnectionError — could not reach the server at all.
        APIError       — server returned an HTTP error (4xx / 5xx).
    """

    def __init__(
        self,
        url:         str           = "http://localhost:8080",
        timeout:     float         = 2.0,
        max_retries: int           = 3,
        api_key:     Optional[str] = None,
    ):
        self.url     = url.rstrip("/")
        self.timeout = timeout

        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST", "DELETE", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if api_key:
            self.session.headers.update({"X-Api-Key": api_key})

    # -- Internal helpers -------------------------------------------------------

    def _get(self, path: str) -> dict:
        try:
            r = self.session.get(f"{self.url}{path}", timeout=self.timeout)
            if r.status_code != 200:
                raise APIError(f"BanditDB {r.status_code}: {r.text}")
            return r.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {path} timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to BanditDB at {self.url}")

    def _post(self, path: str, body: Optional[dict] = None) -> Any:
        try:
            # Only pass json= kwarg when there is a body — keeps call signatures
            # clean for endpoints like /checkpoint that have no request body.
            kwargs: dict = {"timeout": self.timeout}
            if body is not None:
                kwargs["json"] = body
            r = self.session.post(f"{self.url}{path}", **kwargs)
            if r.status_code != 200:
                raise APIError(f"BanditDB {r.status_code}: {r.text}")
            return r.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {path} timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to BanditDB at {self.url}")

    def _delete(self, path: str) -> bool:
        try:
            r = self.session.delete(f"{self.url}{path}", timeout=self.timeout)
            if r.status_code == 404:
                return False
            if r.status_code != 200:
                raise APIError(f"BanditDB {r.status_code}: {r.text}")
            return True
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {path} timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to BanditDB at {self.url}")

    # -- Health -----------------------------------------------------------------

    def health(self) -> bool:
        """
        Return True if the server is reachable and the WAL writer is healthy (HTTP 200).

        Returns True for both ``"ok"`` and ``"degraded"`` overall status — both mean
        the service is available. Returns False only for HTTP 503 (WAL writer failure).
        Use ``health_detail()`` to inspect per-campaign entropy status.

        This endpoint is public and deliberately carries no campaign data. Since
        server 2.0.0 it returns only ``{status, version, features}`` — campaign
        identifiers moved to ``/health/detail`` because they would otherwise expose
        the tenant list to unauthenticated callers.
        """
        try:
            r = self.session.get(f"{self.url}/health", timeout=self.timeout)
            return r.status_code == 200
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Health check timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to BanditDB at {self.url}")

    def server_info(self) -> dict:
        """
        Return ``{status, version, features}`` from the public health endpoint.

        ``features`` lists the Cargo features the server was compiled with. An empty
        list means a plain build: neural algorithms are unavailable and campaigns
        requesting them are rejected at creation.

        Returns ``{}`` for servers older than 2.0.0, which did not report a version.

        Example::

            info = db.server_info()
            if "neural" not in info.get("features", []):
                print("server cannot run neural campaigns")
        """
        try:
            r = self.session.get(f"{self.url}/health", timeout=self.timeout)
            body = r.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Health check timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to BanditDB at {self.url}")
        return {k: body[k] for k in ("status", "version", "features") if k in body}

    def health_detail(self) -> dict:
        """
        Return per-campaign entropy status, scoped to the caller's tenant.

        Requires an API key (reader role or above) and server 2.0.0+, which moved
        this data off the public ``/health`` endpoint.

        Returns a dict with:
            status    : ``"ok"`` | ``"degraded"`` | ``"degraded: wal unavailable"``
            campaigns : mapping of campaign_id →
                            ``{"entropy": float, "status": "ok"|"warning"|"critical"}``

        ``status == "degraded"`` (HTTP 200) means one or more active campaigns have
        low selection entropy without a convergence signal — the WAL is healthy but
        a campaign may have stopped exploring. ``"degraded: wal unavailable"``
        (HTTP 503) means the WAL writer has failed.

        Example::

            detail = db.health_detail()
            for cid, h in detail["campaigns"].items():
                if h["status"] != "ok":
                    print(f"{cid}: entropy={h['entropy']:.2f} ({h['status']})")
        """
        try:
            r = self.session.get(f"{self.url}/health/detail", timeout=self.timeout)
            if r.status_code == 404:
                raise APIError(
                    "GET /health/detail returned 404 — this server predates 2.0.0. "
                    "Older servers return campaign data from /health instead."
                )
            if r.status_code == 401:
                raise APIError("/health/detail requires an API key (reader role or above).")
            return r.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Health check timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to BanditDB at {self.url}")

    # -- Campaigns --------------------------------------------------------------

    def create_campaign(
        self,
        campaign_id:          str,
        arms:                 List[str],
        feature_dim:          int,
        alpha:                float          = 1.0,
        algorithm:            Algorithm      = "linucb",
        metadata:             Optional[dict] = None,
        decay_half_life_hours: Optional[float] = None,
        pacing:               Optional[PacingConfig] = None,
    ) -> bool:
        """
        Create a new decision campaign.

        Parameters
        ----------
        campaign_id           : Unique identifier. ASCII alphanumeric, hyphens, and underscores only.
        arms                  : Decision options (e.g. ["gpt-4o", "claude-haiku", "llama-3"]).
        feature_dim           : Length of the context vector you will pass to predict().
                                For the neural configs this is context_dim, not embed_dim.
        alpha                 : Exploration / exploitation trade-off. Scales the UCB bonus for
                                LinUCB variants and the posterior spread for Thompson
                                Sampling variants. Default 1.0.
        algorithm             : "linucb" | "thompson_sampling" | NeuralLinUCBConfig |
                                NeuralTSConfig | ProgressiveConfig.
        metadata              : Arbitrary JSON dict stored with the campaign (≤ 64 KB).
        decay_half_life_hours : Exponential forgetting. Confidence halves every N hours of
                                real time, measured at each checkpoint. Theta (best estimate)
                                is preserved — only certainty erodes, forcing re-exploration
                                as evidence ages. None = no forgetting (default).
                                Examples: 24.0 (daily drift), 168.0 (weekly), 720.0 (monthly).
        pacing                : Budget constraints (PacingConfig). Server 2.3.0+. Fixed at
                                creation — it cannot be changed or reset afterwards.

        Returns True on success. Raises APIError if the campaign already exists.

        Raises
        ------
        ValueError : pacing is invalid, or an ``arm_costs`` key is not one of ``arms``.
                     The server accepts unknown keys silently, which leaves a
                     misspelt arm costing nothing — checked here instead.
        """
        if pacing is not None:
            pacing._validate()
            known = set(arms)
            for r in pacing.resources:
                unknown = sorted(set(r.arm_costs) - known)
                if unknown:
                    raise ValueError(
                        f"resource '{r.name}' arm_costs names unknown arms {unknown}; "
                        f"campaign arms are {list(arms)}"
                    )
        payload: dict = {
            "campaign_id": campaign_id,
            "arms":        arms,
            "feature_dim": feature_dim,
            "alpha":       alpha,
            "algorithm":   _serialise_algorithm(algorithm),
        }
        if metadata is not None:
            payload["metadata"] = metadata
        if decay_half_life_hours is not None:
            payload["decay_half_life_hours"] = decay_half_life_hours
        if pacing is not None:
            payload["pacing"] = pacing._to_api()
        return self._post("/campaign", payload) == "Campaign Created"

    def list_campaigns(self) -> List[dict]:
        """Return a list of campaign summary dicts (id, alpha, arm_count, archived, algorithm)."""
        return self._get("/campaigns")

    def campaign_info(self, campaign_id: str) -> dict:
        """
        Return the full state for one campaign: per-arm theta vectors, reward counters, etc.
        Pacing campaigns also carry a ``pacing`` key, shaped as pacing() returns.
        Raises APIError (404) if the campaign does not exist.
        """
        return self._get(f"/campaign/{campaign_id}")

    def report(self, campaign_id: str) -> dict:
        """
        Per-arm performance with confidence intervals, plus a ``converged`` flag.

        ``converged`` tests for a **single globally best arm**:
          True   — leading arm has a statistically significant advantage at 95% CI.
          False  — no single global winner yet.
          None   — not enough data yet (< 30 rewards per arm).

        **It stays False on a healthy contextual campaign, by design.** Where
        different arms win for different contexts — push notifications convert on
        mobile, email on desktop — no arm is globally best, so the confidence
        intervals overlap however well the model routes. A campaign operating at
        its theoretical optimum still reports ``converged: False``.

        Read it as "is there one winner for everybody?", not "has this learned?".
        For a contextual campaign, judge learning by the reward-rate trend and by
        whether traffic *splits* across arms — collapsing onto a single arm is the
        signal that the model is not using the context.

        Compare ``arms[arm_id].traffic_share`` to the causal_forest() arm
        assignment percentages (Python SDK ``banditdb.eval.causal_analysis``)
        to verify the bandit has found the causally correct structure.

        Returns
        -------
        dict with keys: campaign_id, total_predictions, total_rewards,
          overall_reward_rate, arms (per-arm stats with CI bounds),
          leading_arm, converged, and optionally challenger_traffic_pct /
          tournament_win_streak for Progressive campaigns.
        """
        return self._get(f"/campaign/{campaign_id}/report")

    def diagnostics(self, campaign_id: str) -> dict:
        """
        Return operator-level diagnostics for a campaign.

        Includes per-arm theta norms, A_inv diagonal bounds (uncertainty proxy),
        tournament traffic percentage and win streak (Progressive campaigns),
        neural replay buffer size (neural campaigns), and entropy alerting fields.

        Entropy alerting fields
        -----------------------
        selection_entropy : float
            Normalised Shannon entropy of arm selection (0 = fully collapsed, 1 = uniform).
        entropy_status : ``"ok"`` | ``"warning"`` | ``"critical"``
            ``"ok"`` when healthy or statistically converged. ``"warning"`` when entropy < 0.4
            without a convergence signal. ``"critical"`` when entropy < 0.2.
        entropy_trend : ``"stable"`` | ``"falling"`` | ``"recovering"`` | ``"unknown"``
            Change since the last checkpoint. ``"falling"`` indicates a recent event
            (pipeline bug, deploy, cohort shift). ``"unknown"`` until first checkpoint.
        converged : bool or None
            Guard 1: True suppresses entropy alerts when one arm has statistically
            won outright. Stays False on contextual campaigns where no arm is
            globally best, even when the model is optimal — there, healthy
            behaviour is high entropy with traffic split by context.
        likely_cause : str or None
            Present when ``entropy_status`` is ``"warning"`` or ``"critical"``.
            One of: ``"recent_collapse"``, ``"early_lock_in"``, ``"sustained_collapse"``.
        suggested_action : str or None
            Remediation guidance. Present alongside ``likely_cause``.

        Use this to answer:
          - Is the model still exploring? (selection_entropy, entropy_status)
          - Did entropy drop recently? (entropy_trend == "falling")
          - Is the model still uncertain about any arm? (a_inv_diag_max still high)
          - Has the Progressive tournament converged? (challenger_traffic_pct stable)
          - Is the neural buffer accumulating data? (neural_buffer_size growing)
        """
        return self._get(f"/campaign/{campaign_id}/diagnostics")

    def pacing(self, campaign_id: str) -> Optional[dict]:
        """
        Budget state of a pacing campaign, or None if it has no pacing. Server 2.3.0+.

        Returns a dict with:
            adaptive  : bool
            resources : list of per-resource dicts —
                name, budget, consumed, remaining, utilization (consumed / budget),
                lambda (current shadow price), horizon, decisions, is_exhausted

        ``lambda`` is what one unit of the resource currently costs an arm's score.
        Rising means spend is running ahead of the target rate; 0 means the budget
        is not binding. Once ``is_exhausted`` is True every arm with a cost for that
        resource stays unselectable — budgets do not reset.

        Example::

            for r in db.pacing("promos")["resources"]:
                print(f"{r['name']}: {r['utilization']:.0%} spent, λ={r['lambda']:.3f}")
        """
        return self._get(f"/campaign/{campaign_id}/pacing")

    def delete_campaign(self, campaign_id: str) -> bool:
        """Permanently delete a campaign. Returns True if deleted, False if not found."""
        return self._delete(f"/campaign/{campaign_id}")

    def archive_campaign(self, campaign_id: str) -> bool:
        """
        Soft-delete a campaign. Archived campaigns reject new predictions and rewards
        but all data (arm matrices, history) is preserved. Recoverable via restore_campaign().
        """
        self._post(f"/campaign/{campaign_id}/archive")
        return True

    def restore_campaign(self, campaign_id: str) -> bool:
        """Restore an archived campaign to active status."""
        self._post(f"/campaign/{campaign_id}/restore")
        return True

    # -- Arms -------------------------------------------------------------------

    def add_arm(
        self,
        campaign_id: str,
        arm_id:      str,
        group:       Optional[str]              = None,
        warm_start:  Optional[dict]             = None,
        costs:       Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Add an arm to a live campaign. Requires an admin key.

        Parameters
        ----------
        campaign_id : Campaign to extend. Must not be archived.
        arm_id      : New arm's id. Must not already exist.
        group       : Hierarchy label. Arms sharing a group can lend their estimate to
                      new members via ``warm_start={"from": "group"}``.
        warm_start  : Prior for the new arm. Default: cold start. One of
                      ``{"from": "population", "strength": 1.0}``,
                      ``{"from": "group", "strength": 1.0}``,
                      ``{"from": "arms", "arms": [...], "strength": 1.0}``.
        costs       : Pacing campaigns only (server 2.3.0+). Units of each resource the
                      arm consumes per prediction, keyed by resource name. Resources
                      left out cost 0. The server rejects unknown resource names and
                      costs on a campaign without pacing.

        Returns True on success. Raises APIError if the arm exists, the campaign is
        archived, or a cost names an unknown resource.

        Raises
        ------
        ValueError : a cost is negative or non-finite.

        Example::

            db.add_arm("promos", "discount_30", costs={"promo_budget": 3.0})
        """
        body: dict = {"arm_id": arm_id}
        if group is not None:
            body["group"] = group
        if warm_start is not None:
            body["warm_start"] = warm_start
        if costs is not None:
            _validate_costs(costs, "costs")
            body["costs"] = costs
        return self._post(f"/campaign/{campaign_id}/arms", body) == "Arm Added"

    def set_arm_status(
        self,
        campaign_id: str,
        arm_id:      str,
        status:      ArmStatus,
    ) -> bool:
        """
        Pause, retire, or reactivate an arm. Requires an admin key.

        Parameters
        ----------
        campaign_id : Campaign the arm belongs to.
        arm_id      : Arm to change.
        status      : ``"active"``  — selectable (the default for every arm).
                      ``"paused"``  — not selectable, keeps learning from in-flight
                                      rewards. Reversible; meant for "out of stock",
                                      "creative paused", "seasonal".
                      ``"retired"`` — not selectable, still absorbs in-flight rewards.
                                      Meant as permanent, but nothing is deleted and
                                      it can be set back to ``"active"``.

        The arm keeps everything it learned in every state. To narrow the candidates
        for a single request without changing state, pass ``eligible_arms`` /
        ``exclude_arms`` to predict() instead.

        Returns True on success. Raises APIError if the campaign or arm does not exist.

        Raises
        ------
        ValueError : status is not one of the three values above.
        """
        if status not in ARM_STATUSES:
            raise ValueError(f"status must be one of {list(ARM_STATUSES)}, got {status!r}")
        return self._post(
            f"/campaign/{campaign_id}/arms/{arm_id}/status", {"status": status}
        ) == "Arm Status Updated"

    # -- Predict / Reward -------------------------------------------------------

    def predict(
        self,
        campaign_id:   str,
        context:       List[float],
        eligible_arms: Optional[List[str]] = None,
        exclude_arms:  Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """
        Select the best arm for the given context.

        Returns (arm_id, interaction_id). You MUST pass interaction_id to reward()
        to close the learning loop.

        Parameters
        ----------
        campaign_id : Which campaign to query.
        context     : Feature vector. Length must match the campaign's feature_dim.
                      Values must be finite and within ±1e6. Normalise to unit L2
                      norm — see normalize_context() for why it matters.
        eligible_arms : Restrict this request to these arms. None = every active arm.
        exclude_arms  : Drop these arms from this request, applied after
                        ``eligible_arms``. Use for per-request constraints — out of
                        stock for this shopper, already shown to this user. Neither
                        filter changes campaign state; see set_arm_status() for that.
                        If nothing is left to choose from, the server returns an error.

        Returns
        -------
        (arm_id, interaction_id)

        Raises
        ------
        ValueError : context is empty, non-finite, or too large. Checked locally so
                     you get the offending index rather than a 400 from the server.
        """
        _validate_context(context)
        body: dict = {"campaign_id": campaign_id, "context": context}
        if eligible_arms is not None:
            body["eligible_arms"] = eligible_arms
        if exclude_arms is not None:
            body["exclude_arms"] = exclude_arms
        data = self._post("/predict", body)
        return data["arm_id"], data["interaction_id"]

    def batch_predict(
        self,
        predictions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Predict for up to 100 campaign/context pairs in a single round-trip.

        Each item in `predictions` must have keys:
            campaign_id : str
            context     : List[float]
        and may have ``eligible_arms`` / ``exclude_arms``, as in predict().

        Returns a list of result dicts. Each dict has either:
            arm_id, interaction_id  — on success
            error                   — on per-item failure (other items still succeed)

        Example
        -------
        results = db.batch_predict([
            {"campaign_id": "prices", "context": [0.3, 0.7]},
            {"campaign_id": "layout", "context": [0.1, 0.5]},
        ])
        for r in results:
            if "error" not in r:
                print(r["arm_id"], r["interaction_id"])

        Raises
        ------
        ValueError : any item is missing a key or carries an invalid context. The
                     whole batch is checked before sending, so one bad item does not
                     consume a round trip.
        """
        for i, item in enumerate(predictions):
            if "campaign_id" not in item or "context" not in item:
                raise ValueError(f"predictions[{i}] needs both 'campaign_id' and 'context'")
            try:
                _validate_context(item["context"])
            except ValueError as e:
                raise ValueError(f"predictions[{i}]: {e}") from None
        return self._post("/batch_predict", {"predictions": predictions})

    def reward(self, interaction_id: str, reward: float) -> bool:
        """
        Record the observed reward for a prediction.

        Parameters
        ----------
        interaction_id : Returned by predict() or batch_predict().
        reward         : Observed outcome in [0.0, 1.0]. Values outside the range are
                         rejected — rescale instead (e.g. revenue / max_revenue).

        Returns True on success. Raises APIError if the interaction_id has
        already been rewarded or has expired (default TTL: 24 hours).

        **This call waits for the server to fsync.** Since 2.0.0 a success response
        means the reward is on disk and survives power loss, which costs roughly
        3.4 ms. The server amortises the fsync across concurrent callers, so
        submitting rewards in parallel reaches far higher throughput (~4,400/s) than
        a serial loop. Batch or thread if throughput matters.

        Raises
        ------
        ValueError : reward is non-finite or outside [0.0, 1.0].
        """
        _validate_reward(reward)
        return self._post("/reward", {"interaction_id": interaction_id, "reward": reward}) == "OK"

    def interact(
        self,
        campaign_id: str,
        arm_id: str,
        context: List[float],
        reward: float,
    ) -> str:
        """
        Record a decision that was made elsewhere, together with its outcome.

        For backfilling a model from existing logs: you already know which arm was
        shown and what happened, so there is nothing to predict. Both records are
        written durably before this returns.

        Parameters
        ----------
        campaign_id : Must already exist.
        arm_id      : The arm that was actually shown.
        context     : Feature vector, same constraints as predict().
        reward      : Observed outcome in [0.0, 1.0].

        Returns the interaction_id assigned to the backfilled record.

        Example::

            for row in historical_log:
                db.interact("prices", row.variant, row.features, row.converted)
        """
        _validate_context(context)
        _validate_reward(reward)
        data = self._post(
            f"/campaign/{campaign_id}/interact",
            {"arm_id": arm_id, "context": context, "reward": reward},
        )
        return data["interaction_id"]

    # -- Checkpoint / Export ----------------------------------------------------

    def checkpoint(self) -> str:
        """
        Flush the WAL, snapshot all campaign matrices, write Parquet shards,
        run neural retrain + tournament evaluation (if applicable), and rotate the WAL.

        Returns a summary string. Call this on a schedule or trigger it when
        BANDITDB_MAX_WAL_SIZE_MB or BANDITDB_CHECKPOINT_INTERVAL is hit.
        """
        return self._post("/checkpoint")

    def export(self) -> dict:
        """
        List Parquet export shards, grouped by campaign and sorted chronologically.

        Returns a dict with keys:
            export_dir : str
            shards     : { campaign_id: [filename, ...] }

        Load a shard for offline analysis:
            import polars as pl
            from banditdb.eval import doubly_robust
            df = pl.read_parquet("/data/exports/prices_1234567890.parquet")
            print(doubly_robust(df))
        """
        return self._get("/export")
