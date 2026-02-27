import logging
from typing import List, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import ConnectionError, TimeoutError, APIError

logger = logging.getLogger(__name__)

class Client:
    """Production-ready synchronous client for BanditDB."""
    
    def __init__(self, url: str = "http://localhost:8080", timeout: float = 2.0, max_retries: int = 3):
        self.url = url.rstrip("/")
        self.timeout = timeout
        
        # Configure robust connection pooling and automatic retries
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=0.1,  # 0.1s, 0.2s, 0.4s between retries
            status_forcelist=[500, 502, 503, 504], # Retry on server errors
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def predict(self, campaign_id: str, context: List[float]) -> Tuple[str, str]:
        """
        Ask the database which arm to choose based on context.
        Returns: (arm_id, interaction_id)
        """
        try:
            response = self.session.post(
                f"{self.url}/predict",
                json={"campaign_id": campaign_id, "context": context},
                timeout=self.timeout
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
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise APIError(f"BanditDB Error: {response.text}")
                
            return response.json() == "OK"
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"BanditDB reward timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to BanditDB")
            
    def create_campaign(self, campaign_id: str, arms: List[str], feature_dim: int) -> bool:
        """Create a new Multi-Armed Bandit campaign dynamically."""
        try:
            response = self.session.post(
                f"{self.url}/campaign",
                json={
                    "campaign_id": campaign_id,
                    "arms": arms,
                    "feature_dim": feature_dim
                },
                timeout=self.timeout
            )
            if response.status_code != 200:
                raise APIError(f"BanditDB Error: {response.text}")
            return response.json() == "Campaign Created"
            
        except requests.exceptions.Timeout:
            raise TimeoutError("BanditDB request timed out.")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to BanditDB")
