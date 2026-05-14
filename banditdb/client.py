"""
banditdb.client — Synchronous Python client for BanditDB.

Quickstart:
    from banditdb import Client, NeuralLinUCBConfig, ProgressiveConfig

    db = Client("http://localhost:8080", api_key="...")

    # Simple LinUCB campaign
    db.create_campaign("prices", ["10", "15", "20"], feature_dim=5)

    arm, iid = db.predict("prices", [0.3, 0.7, 0.1, 0.9, 0.4])
    db.reward(iid, 1.0)

    # Neural contextual bandit
    cfg = NeuralLinUCBConfig(context_dim=5, embed_dim=32)
    db.create_campaign("prices_neural", ["10", "15", "20"], feature_dim=5, algorithm=cfg)

    # Self-tuning tournament (LinUCB vs NeuralLinUCB)
    cfg = ProgressiveConfig(
        base="linucb",
        challenger=NeuralLinUCBConfig(context_dim=5, embed_dim=32),
    )
    db.create_campaign("prices_prog", ["10", "15", "20"], feature_dim=5, algorithm=cfg)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import APIError, ConnectionError, TimeoutError

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

    def _to_api(self) -> dict:
        return {
            "neural_lin_ucb": {
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
    base          : Base algorithm. "linucb", "thompson_sampling", or NeuralLinUCBConfig.
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
    ProgressiveConfig,
]

def _serialise_algorithm(algo: Algorithm) -> Any:
    if isinstance(algo, str):
        return algo
    return algo._to_api()

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
        """
        try:
            r = self.session.get(f"{self.url}/health", timeout=self.timeout)
            return r.status_code == 200
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Health check timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to BanditDB at {self.url}")

    def health_detail(self) -> dict:
        """
        Return the full health response including per-campaign entropy status.

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
            r = self.session.get(f"{self.url}/health", timeout=self.timeout)
            return r.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Health check timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to BanditDB at {self.url}")

    # -- Campaigns --------------------------------------------------------------

    def create_campaign(
        self,
        campaign_id: str,
        arms:        List[str],
        feature_dim: int,
        alpha:       float                = 1.0,
        algorithm:   Algorithm            = "linucb",
        metadata:    Optional[dict]       = None,
    ) -> bool:
        """
        Create a new decision campaign.

        Parameters
        ----------
        campaign_id : Unique identifier. ASCII alphanumeric, hyphens, and underscores only.
        arms        : Decision options (e.g. ["gpt-4o", "claude-haiku", "llama-3"]).
        feature_dim : Length of the context vector you will pass to predict().
                      For NeuralLinUCBConfig this is context_dim, not embed_dim.
        alpha       : Exploration / exploitation trade-off (LinUCB and TS). Default 1.0.
        algorithm   : "linucb" | "thompson_sampling" | NeuralLinUCBConfig | ProgressiveConfig.
        metadata    : Arbitrary JSON dict stored with the campaign (≤ 64 KB).

        Returns True on success. Raises APIError if the campaign already exists.
        """
        payload: dict = {
            "campaign_id": campaign_id,
            "arms":        arms,
            "feature_dim": feature_dim,
            "alpha":       alpha,
            "algorithm":   _serialise_algorithm(algorithm),
        }
        if metadata is not None:
            payload["metadata"] = metadata
        return self._post("/campaign", payload) == "Campaign Created"

    def list_campaigns(self) -> List[dict]:
        """Return a list of campaign summary dicts (id, alpha, arm_count, archived, algorithm)."""
        return self._get("/campaigns")

    def campaign_info(self, campaign_id: str) -> dict:
        """
        Return the full state for one campaign: per-arm theta vectors, reward counters, etc.
        Raises APIError (404) if the campaign does not exist.
        """
        return self._get(f"/campaign/{campaign_id}")

    def report(self, campaign_id: str) -> dict:
        """
        Get the business-level convergence report for a campaign.

        The ``converged`` field answers "is this campaign done?":
          True   — leading arm has a statistically significant advantage at 95% CI.
          False  — leading but CIs still overlap — keep collecting data.
          None   — not enough data yet (< 30 rewards per arm).

        Compare ``arms[arm_id].traffic_share`` to the causal_forest() arm
        assignment percentages (Python SDK ``banditdb.eval.causal_analysis``)
        to verify the bandit has converged to the causally correct structure.

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
        neural replay buffer size (NeuralLinUCB campaigns), and entropy alerting fields.

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
            Guard 1: True suppresses entropy alerts when one arm has statistically won.
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

    # -- Predict / Reward -------------------------------------------------------

    def predict(self, campaign_id: str, context: List[float]) -> Tuple[str, str]:
        """
        Select the best arm for the given context.

        Returns (arm_id, interaction_id). You MUST pass interaction_id to reward()
        to close the learning loop.

        Parameters
        ----------
        campaign_id : Which campaign to query.
        context     : Feature vector. Length must match the campaign's feature_dim.

        Returns
        -------
        (arm_id, interaction_id)
        """
        data = self._post("/predict", {"campaign_id": campaign_id, "context": context})
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
        """
        return self._post("/batch_predict", {"predictions": predictions})

    def reward(self, interaction_id: str, reward: float) -> bool:
        """
        Record the observed reward for a prediction.

        Parameters
        ----------
        interaction_id : Returned by predict() or batch_predict().
        reward         : Observed outcome in [0.0, 1.0].

        Returns True on success. Raises APIError if the interaction_id has
        already been rewarded or has expired (default TTL: 24 hours).
        """
        return self._post("/reward", {"interaction_id": interaction_id, "reward": reward}) == "OK"

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
