import logging
import requests
from typing import List, Literal, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import ConnectionError, TimeoutError, APIError

logger = logging.getLogger(__name__)

class Client:
    """Production-ready synchronous client for BanditDB."""

    def __init__(
        self,
        url: str = "http://localhost:8080",
        timeout: float = 2.0,
        max_retries: int = 3,
        api_key: Optional[str] = None,
    ):
        self.url = url.rstrip("/")
        self.timeout = timeout

        # Configure robust connection pooling and automatic retries
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=0.1,  # 0.1s, 0.2s, 0.4s between retries
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST", "DELETE", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set the API key header once on the session so every request carries it
        if api_key:
            self.session.headers.update({"X-Api-Key": api_key})

    def health(self) -> bool:
        """Return True if the server is reachable and healthy."""
        try:
            response = self.session.get(f"{self.url}/health", timeout=self.timeout)
            return response.status_code == 200
        except requests.exceptions.Timeout:
            raise TimeoutError(f"BanditDB health check timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to BanditDB at {self.url}")

    def create_campaign(
        self,
        campaign_id: str,
        arms: List[str],
        feature_dim: int,
        alpha: float = 1.0,
        algorithm: Literal["linucb", "thompson_sampling"] = "linucb",
    ) -> bool:
        """
        Create a new Multi-Armed Bandit campaign.

        alpha controls the exploration/exploitation trade-off.
        Lower values (e.g. 0.1) exploit learned knowledge more aggressively.
        Higher values (e.g. 3.0) keep exploring uncertain arms longer.
        Defaults to 1.0.

        algorithm selects the decision algorithm:
          "linucb" (default) — deterministic UCB bonus; tune alpha to control exploration.
          "thompson_sampling" — samples from the Bayesian posterior N(θ, alpha²·A⁻¹);
            alpha=1.0 is the principled default (natural posterior width). Concurrent
            users automatically diversify arm coverage with no alpha sweep needed.
        """
        try:
            response = self.session.post(
                f"{self.url}/campaign",
                json={
                    "campaign_id": campaign_id,
                    "arms": arms,
                    "feature_dim": feature_dim,
                    "alpha": alpha,
                    "algorithm": algorithm,
                },
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise APIError(f"BanditDB Error: {response.text}")
            return response.json() == "Campaign Created"
        except requests.exceptions.Timeout:
            raise TimeoutError("BanditDB request timed out.")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to BanditDB")

    def list_campaigns(self) -> List[dict]:
        """
        List all live campaigns.
        Returns a list of dicts with keys: campaign_id, alpha, arm_count.
        """
        try:
            response = self.session.get(f"{self.url}/campaigns", timeout=self.timeout)
            if response.status_code != 200:
                raise APIError(f"BanditDB Error: {response.text}")
            return response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"BanditDB request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to BanditDB")

    def campaign_info(self, campaign_id: str) -> dict:
        """
        Return the full diagnostic state for one campaign.

        The returned dict contains:
          campaign_id, alpha, total_predictions, total_rewards,
          arms: { arm_id: { theta, theta_norm, prediction_count, reward_count } }

        Raises APIError (404) if the campaign does not exist.
        """
        try:
            response = self.session.get(
                f"{self.url}/campaign/{campaign_id}", timeout=self.timeout
            )
            if response.status_code != 200:
                raise APIError(f"BanditDB Error: {response.text}")
            return response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"BanditDB request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to BanditDB")

    def delete_campaign(self, campaign_id: str) -> bool:
        """Delete a campaign. Returns True if deleted, False if not found."""
        try:
            response = self.session.delete(
                f"{self.url}/campaign/{campaign_id}",
                timeout=self.timeout,
            )
            if response.status_code == 404:
                return False
            if response.status_code != 200:
                raise APIError(f"BanditDB Error: {response.text}")
            return True
        except requests.exceptions.Timeout:
            raise TimeoutError(f"BanditDB request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to BanditDB")

    def predict(self, campaign_id: str, context: List[float]) -> Tuple[str, str]:
        """
        Ask the database which arm to choose based on context.
        Returns: (arm_id, interaction_id)
        """
        try:
            response = self.session.post(
                f"{self.url}/predict",
                json={"campaign_id": campaign_id, "context": context},
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise APIError(f"BanditDB Error: {response.text}")
            data = response.json()
            return data["arm_id"], data["interaction_id"]
        except requests.exceptions.Timeout:
            raise TimeoutError(f"BanditDB request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to BanditDB at {self.url}")

    def reward(self, interaction_id: str, reward: float) -> bool:
        """
        Send a reward back to the database.
        Returns True if successful.
        """
        try:
            response = self.session.post(
                f"{self.url}/reward",
                json={"interaction_id": interaction_id, "reward": reward},
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise APIError(f"BanditDB Error: {response.text}")
            return response.json() == "OK"
        except requests.exceptions.Timeout:
            raise TimeoutError(f"BanditDB reward timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to BanditDB")

    def checkpoint(self) -> str:
        """
        Flush the WAL, snapshot all campaign matrices, write completed
        prediction→reward pairs to per-campaign Parquet files, and rotate
        the WAL. Returns a summary string from the server.

        Call this on a schedule or after significant traffic to keep the
        WAL small and the Parquet export up to date.
        """
        try:
            response = self.session.post(f"{self.url}/checkpoint", timeout=self.timeout)
            if response.status_code != 200:
                raise APIError(f"BanditDB Error: {response.text}")
            return response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"BanditDB checkpoint timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to BanditDB")

    def export(self) -> str:
        """
        List the per-campaign Parquet files available in the server's exports
        directory. Files are created by checkpoint().

        Returns a formatted string, e.g.:
          'Parquet files in /data/exports: ["llm_routing.parquet"]'
        """
        try:
            response = self.session.get(f"{self.url}/export", timeout=self.timeout)
            if response.status_code != 200:
                raise APIError(f"BanditDB Error: {response.text}")
            return response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"BanditDB export timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to BanditDB")
