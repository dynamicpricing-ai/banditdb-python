"""
Unit tests for ``banditdb.Client``
===================================

Coverage
--------
``Client.__init__``
    URL normalisation, API-key header injection, custom timeout storage,
    absence of auth header when no key is provided.

``Client.health``
    Healthy server (200), degraded server (non-200), timeout, unreachable host.

``Client.create_campaign``
    Successful creation, correct JSON payload, HTTP error codes (parametrised),
    error text propagation, timeout, unreachable host.

``Client.delete_campaign``
    Campaign deleted (200), campaign not found (404), HTTP error codes
    (parametrised), correct endpoint construction, timeout, unreachable host.

``Client.predict``
    Arm-ID / interaction-ID unpacking, correct JSON payload, HTTP error codes
    (parametrised), error text propagation, timeout, unreachable host.

``Client.reward``
    Successful reward acknowledgement, correct JSON payload including a zero
    reward, HTTP error codes (parametrised), error text propagation, timeout,
    unreachable host.

``Client.export``
    Confirmation-message passthrough, correct endpoint, HTTP error codes
    (parametrised), timeout, unreachable host.

Design notes
------------
* Tests are grouped into one ``TestCase`` class per client method so that
  ``pytest -v`` output reads as a structured specification.
* Every test mocks at the ``requests.Session`` level — no live server needed.
* ``make_response`` and the ``client`` fixture are provided by ``conftest.py``.
* ``pytest.mark.parametrize`` is used wherever multiple HTTP error codes must
  all trigger the same exception, avoiding repetitive boilerplate.
"""

import re

import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

from banditdb import Client
from banditdb.exceptions import APIError, ConnectionError, TimeoutError

from tests.conftest import BASE_URL, API_KEY, make_response


# ══════════════════════════════════════════════════════════════════════════════
# Client.__init__
# ══════════════════════════════════════════════════════════════════════════════

class TestClientInitialisation:
    """Verifies that ``__init__`` stores configuration correctly."""

    def test_default_url_is_localhost(self):
        c = Client()
        assert c.url == "http://localhost:8080"

    def test_trailing_slash_is_stripped_from_url(self):
        c = Client(url="http://localhost:8080/")
        assert c.url == "http://localhost:8080"

    def test_custom_timeout_is_stored(self):
        c = Client(timeout=5.0)
        assert c.timeout == 5.0

    def test_api_key_is_injected_into_session_headers(self):
        c = Client(api_key="my-secret-key")
        assert c.session.headers["X-Api-Key"] == "my-secret-key"

    def test_no_api_key_omits_auth_header(self):
        c = Client()
        assert "X-Api-Key" not in c.session.headers


# ══════════════════════════════════════════════════════════════════════════════
# Client.health
# ══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    """Verifies ``GET /health`` behaviour across all code paths."""

    def test_returns_true_when_server_is_healthy(self, client):
        client.session.get.return_value = make_response(200, {"status": "ok"})
        assert client.health() is True

    def test_returns_false_when_server_is_degraded(self, client):
        client.session.get.return_value = make_response(503)
        assert client.health() is False

    def test_calls_correct_endpoint(self, client):
        client.session.get.return_value = make_response(200)
        client.health()
        client.session.get.assert_called_once_with(
            f"{BASE_URL}/health", timeout=2.0
        )

    def test_raises_timeout_error_when_request_times_out(self, client):
        client.session.get.side_effect = RequestsTimeout()
        with pytest.raises(TimeoutError, match="timed out"):
            client.health()

    def test_raises_connection_error_when_server_is_unreachable(self, client):
        client.session.get.side_effect = RequestsConnectionError()
        with pytest.raises(ConnectionError, match="Failed to connect"):
            client.health()


# ══════════════════════════════════════════════════════════════════════════════
# Client.create_campaign
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateCampaign:
    """Verifies ``POST /campaign`` behaviour across all code paths."""

    CAMPAIGN_ID = "llm_routing"
    ARMS        = ["gpt-4o", "claude-haiku", "llama-3"]
    FEATURE_DIM = 3

    def test_returns_true_on_successful_creation(self, client):
        client.session.post.return_value = make_response(200, "Campaign Created")
        result = client.create_campaign(self.CAMPAIGN_ID, self.ARMS, self.FEATURE_DIM)
        assert result is True

    def test_sends_correct_json_payload(self, client):
        client.session.post.return_value = make_response(200, "Campaign Created")
        client.create_campaign(self.CAMPAIGN_ID, self.ARMS, self.FEATURE_DIM)
        client.session.post.assert_called_once_with(
            f"{BASE_URL}/campaign",
            json={
                "campaign_id": self.CAMPAIGN_ID,
                "arms":        self.ARMS,
                "feature_dim": self.FEATURE_DIM,
                "alpha":       1.0,
            },
            timeout=2.0,
        )

    def test_sends_custom_alpha_in_payload(self, client):
        client.session.post.return_value = make_response(200, "Campaign Created")
        client.create_campaign(self.CAMPAIGN_ID, self.ARMS, self.FEATURE_DIM, alpha=2.5)
        client.session.post.assert_called_once_with(
            f"{BASE_URL}/campaign",
            json={
                "campaign_id": self.CAMPAIGN_ID,
                "arms":        self.ARMS,
                "feature_dim": self.FEATURE_DIM,
                "alpha":       2.5,
            },
            timeout=2.0,
        )

    @pytest.mark.parametrize("status_code", [400, 409, 422, 500])
    def test_raises_api_error_on_http_error(self, client, status_code):
        client.session.post.return_value = make_response(
            status_code, text="campaign already exists"
        )
        with pytest.raises(APIError):
            client.create_campaign(self.CAMPAIGN_ID, self.ARMS, self.FEATURE_DIM)

    def test_api_error_message_includes_server_response(self, client):
        server_msg = "feature_dim must be greater than 0"
        client.session.post.return_value = make_response(400, text=server_msg)
        with pytest.raises(APIError, match=server_msg):
            client.create_campaign(self.CAMPAIGN_ID, self.ARMS, self.FEATURE_DIM)

    def test_raises_timeout_error_when_request_times_out(self, client):
        client.session.post.side_effect = RequestsTimeout()
        with pytest.raises(TimeoutError):
            client.create_campaign(self.CAMPAIGN_ID, self.ARMS, self.FEATURE_DIM)

    def test_raises_connection_error_when_server_is_unreachable(self, client):
        client.session.post.side_effect = RequestsConnectionError()
        with pytest.raises(ConnectionError):
            client.create_campaign(self.CAMPAIGN_ID, self.ARMS, self.FEATURE_DIM)


# ══════════════════════════════════════════════════════════════════════════════
# Client.delete_campaign
# ══════════════════════════════════════════════════════════════════════════════

class TestDeleteCampaign:
    """Verifies ``DELETE /campaign/:id`` behaviour across all code paths."""

    CAMPAIGN_ID = "llm_routing"

    def test_returns_true_when_campaign_is_deleted(self, client):
        client.session.delete.return_value = make_response(200)
        assert client.delete_campaign(self.CAMPAIGN_ID) is True

    def test_returns_false_when_campaign_not_found(self, client):
        client.session.delete.return_value = make_response(404, text="not found")
        assert client.delete_campaign(self.CAMPAIGN_ID) is False

    def test_calls_correct_endpoint(self, client):
        client.session.delete.return_value = make_response(200)
        client.delete_campaign(self.CAMPAIGN_ID)
        client.session.delete.assert_called_once_with(
            f"{BASE_URL}/campaign/{self.CAMPAIGN_ID}",
            timeout=2.0,
        )

    @pytest.mark.parametrize("status_code", [400, 401, 500, 503])
    def test_raises_api_error_on_http_error(self, client, status_code):
        client.session.delete.return_value = make_response(
            status_code, text="server error"
        )
        with pytest.raises(APIError):
            client.delete_campaign(self.CAMPAIGN_ID)

    def test_raises_timeout_error_when_request_times_out(self, client):
        client.session.delete.side_effect = RequestsTimeout()
        with pytest.raises(TimeoutError):
            client.delete_campaign(self.CAMPAIGN_ID)

    def test_raises_connection_error_when_server_is_unreachable(self, client):
        client.session.delete.side_effect = RequestsConnectionError()
        with pytest.raises(ConnectionError):
            client.delete_campaign(self.CAMPAIGN_ID)


# ══════════════════════════════════════════════════════════════════════════════
# Client.predict
# ══════════════════════════════════════════════════════════════════════════════

class TestPredict:
    """Verifies ``POST /predict`` behaviour across all code paths."""

    CAMPAIGN_ID    = "llm_routing"
    CONTEXT        = [0.2, 0.9, 0.1]
    ARM_ID         = "claude-haiku"
    INTERACTION_ID = "01JNVK38P4QBZT6X8YEKN2MFR1"

    def _success_response(self):
        return make_response(
            200,
            {"arm_id": self.ARM_ID, "interaction_id": self.INTERACTION_ID},
        )

    def test_returns_arm_id_and_interaction_id(self, client):
        client.session.post.return_value = self._success_response()
        arm_id, interaction_id = client.predict(self.CAMPAIGN_ID, self.CONTEXT)
        assert arm_id == self.ARM_ID
        assert interaction_id == self.INTERACTION_ID

    def test_sends_correct_json_payload(self, client):
        client.session.post.return_value = self._success_response()
        client.predict(self.CAMPAIGN_ID, self.CONTEXT)
        client.session.post.assert_called_once_with(
            f"{BASE_URL}/predict",
            json={"campaign_id": self.CAMPAIGN_ID, "context": self.CONTEXT},
            timeout=2.0,
        )

    @pytest.mark.parametrize("status_code", [400, 404, 422, 500])
    def test_raises_api_error_on_http_error(self, client, status_code):
        client.session.post.return_value = make_response(
            status_code, text="campaign not found"
        )
        with pytest.raises(APIError):
            client.predict(self.CAMPAIGN_ID, self.CONTEXT)

    def test_api_error_message_includes_server_response(self, client):
        server_msg = "context vector has wrong dimension: expected 3, got 5"
        client.session.post.return_value = make_response(400, text=server_msg)
        with pytest.raises(APIError, match=server_msg):
            client.predict(self.CAMPAIGN_ID, self.CONTEXT)

    def test_raises_timeout_error_when_request_times_out(self, client):
        client.session.post.side_effect = RequestsTimeout()
        with pytest.raises(TimeoutError):
            client.predict(self.CAMPAIGN_ID, self.CONTEXT)

    def test_raises_connection_error_when_server_is_unreachable(self, client):
        client.session.post.side_effect = RequestsConnectionError()
        with pytest.raises(ConnectionError):
            client.predict(self.CAMPAIGN_ID, self.CONTEXT)


# ══════════════════════════════════════════════════════════════════════════════
# Client.reward
# ══════════════════════════════════════════════════════════════════════════════

class TestReward:
    """Verifies ``POST /reward`` behaviour across all code paths."""

    INTERACTION_ID = "01JNVK38P4QBZT6X8YEKN2MFR1"
    REWARD         = 1.0

    def test_returns_true_on_acknowledged_reward(self, client):
        client.session.post.return_value = make_response(200, "OK")
        assert client.reward(self.INTERACTION_ID, self.REWARD) is True

    def test_sends_correct_json_payload(self, client):
        client.session.post.return_value = make_response(200, "OK")
        client.reward(self.INTERACTION_ID, self.REWARD)
        client.session.post.assert_called_once_with(
            f"{BASE_URL}/reward",
            json={"interaction_id": self.INTERACTION_ID, "reward": self.REWARD},
            timeout=2.0,
        )

    def test_sends_zero_reward_without_coercion(self, client):
        """Ensure 0.0 is transmitted as-is and not silently dropped or converted."""
        client.session.post.return_value = make_response(200, "OK")
        client.reward(self.INTERACTION_ID, 0.0)
        client.session.post.assert_called_once_with(
            f"{BASE_URL}/reward",
            json={"interaction_id": self.INTERACTION_ID, "reward": 0.0},
            timeout=2.0,
        )

    @pytest.mark.parametrize("status_code", [404, 410, 422, 500])
    def test_raises_api_error_on_http_error(self, client, status_code):
        client.session.post.return_value = make_response(
            status_code, text="interaction not found or expired"
        )
        with pytest.raises(APIError):
            client.reward(self.INTERACTION_ID, self.REWARD)

    def test_api_error_message_includes_server_response(self, client):
        server_msg = "reward must be in [0, 1], got 1.5"
        client.session.post.return_value = make_response(422, text=server_msg)
        with pytest.raises(APIError, match=re.escape(server_msg)):
            client.reward(self.INTERACTION_ID, self.REWARD)

    def test_raises_timeout_error_when_request_times_out(self, client):
        client.session.post.side_effect = RequestsTimeout()
        with pytest.raises(TimeoutError):
            client.reward(self.INTERACTION_ID, self.REWARD)

    def test_raises_connection_error_when_server_is_unreachable(self, client):
        client.session.post.side_effect = RequestsConnectionError()
        with pytest.raises(ConnectionError):
            client.reward(self.INTERACTION_ID, self.REWARD)


# ══════════════════════════════════════════════════════════════════════════════
# Client.list_campaigns
# ══════════════════════════════════════════════════════════════════════════════

class TestListCampaigns:
    """Verifies ``GET /campaigns`` behaviour across all code paths."""

    CAMPAIGNS = [
        {"campaign_id": "llm_routing", "alpha": 1.0, "arm_count": 3},
        {"campaign_id": "pricing",     "alpha": 0.5, "arm_count": 2},
    ]

    def test_returns_list_of_campaign_summaries(self, client):
        client.session.get.return_value = make_response(200, self.CAMPAIGNS)
        assert client.list_campaigns() == self.CAMPAIGNS

    def test_calls_correct_endpoint(self, client):
        client.session.get.return_value = make_response(200, self.CAMPAIGNS)
        client.list_campaigns()
        client.session.get.assert_called_once_with(
            f"{BASE_URL}/campaigns", timeout=2.0
        )

    @pytest.mark.parametrize("status_code", [401, 500, 503])
    def test_raises_api_error_on_http_error(self, client, status_code):
        client.session.get.return_value = make_response(status_code, text="server error")
        with pytest.raises(APIError):
            client.list_campaigns()

    def test_raises_timeout_error_when_request_times_out(self, client):
        client.session.get.side_effect = RequestsTimeout()
        with pytest.raises(TimeoutError):
            client.list_campaigns()

    def test_raises_connection_error_when_server_is_unreachable(self, client):
        client.session.get.side_effect = RequestsConnectionError()
        with pytest.raises(ConnectionError):
            client.list_campaigns()


# ══════════════════════════════════════════════════════════════════════════════
# Client.campaign_info
# ══════════════════════════════════════════════════════════════════════════════

class TestCampaignInfo:
    """Verifies ``GET /campaign/:id`` behaviour across all code paths."""

    CAMPAIGN_ID = "llm_routing"
    INFO = {
        "campaign_id": "llm_routing",
        "alpha": 1.0,
        "total_predictions": 2415,
        "total_rewards": 2255,
        "arms": {
            "gpt-4o": {
                "theta": [0.31, -0.08, 0.72],
                "theta_norm": 0.784,
                "prediction_count": 1523,
                "reward_count": 1421,
            },
            "claude-haiku": {
                "theta": [0.12, 0.41, 0.19],
                "theta_norm": 0.456,
                "prediction_count": 892,
                "reward_count": 834,
            },
        },
    }

    def test_returns_campaign_diagnostic(self, client):
        client.session.get.return_value = make_response(200, self.INFO)
        result = client.campaign_info(self.CAMPAIGN_ID)
        assert result["campaign_id"] == self.CAMPAIGN_ID
        assert "arms" in result
        assert "total_predictions" in result

    def test_calls_correct_endpoint(self, client):
        client.session.get.return_value = make_response(200, self.INFO)
        client.campaign_info(self.CAMPAIGN_ID)
        client.session.get.assert_called_once_with(
            f"{BASE_URL}/campaign/{self.CAMPAIGN_ID}", timeout=2.0
        )

    def test_raises_api_error_when_campaign_not_found(self, client):
        client.session.get.return_value = make_response(
            404, text="Campaign 'nonexistent' not found"
        )
        with pytest.raises(APIError):
            client.campaign_info("nonexistent")

    @pytest.mark.parametrize("status_code", [401, 500, 503])
    def test_raises_api_error_on_http_error(self, client, status_code):
        client.session.get.return_value = make_response(status_code, text="server error")
        with pytest.raises(APIError):
            client.campaign_info(self.CAMPAIGN_ID)

    def test_raises_timeout_error_when_request_times_out(self, client):
        client.session.get.side_effect = RequestsTimeout()
        with pytest.raises(TimeoutError):
            client.campaign_info(self.CAMPAIGN_ID)

    def test_raises_connection_error_when_server_is_unreachable(self, client):
        client.session.get.side_effect = RequestsConnectionError()
        with pytest.raises(ConnectionError):
            client.campaign_info(self.CAMPAIGN_ID)


# ══════════════════════════════════════════════════════════════════════════════
# Client.checkpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpoint:
    """Verifies ``POST /checkpoint`` behaviour across all code paths."""

    SUMMARY = (
        "Checkpoint written and WAL rotated: 2 campaigns, offset 4821 bytes, "
        "150 interactions exported, 0 in-flight re-emitted"
    )

    def test_returns_summary_string(self, client):
        client.session.post.return_value = make_response(200, self.SUMMARY)
        assert client.checkpoint() == self.SUMMARY

    def test_calls_correct_endpoint(self, client):
        client.session.post.return_value = make_response(200, self.SUMMARY)
        client.checkpoint()
        client.session.post.assert_called_once_with(
            f"{BASE_URL}/checkpoint", timeout=2.0
        )

    @pytest.mark.parametrize("status_code", [401, 500, 503])
    def test_raises_api_error_on_http_error(self, client, status_code):
        client.session.post.return_value = make_response(
            status_code, text="checkpoint failed"
        )
        with pytest.raises(APIError):
            client.checkpoint()

    def test_raises_timeout_error_when_request_times_out(self, client):
        client.session.post.side_effect = RequestsTimeout()
        with pytest.raises(TimeoutError):
            client.checkpoint()

    def test_raises_connection_error_when_server_is_unreachable(self, client):
        client.session.post.side_effect = RequestsConnectionError()
        with pytest.raises(ConnectionError):
            client.checkpoint()


# ══════════════════════════════════════════════════════════════════════════════
# Client.export
# ══════════════════════════════════════════════════════════════════════════════

class TestExport:
    """Verifies ``GET /export`` behaviour across all code paths."""

    CONFIRMATION = 'Parquet files in /data/exports: ["llm_routing.parquet"]'

    def test_returns_confirmation_message_on_success(self, client):
        client.session.get.return_value = make_response(200, self.CONFIRMATION)
        assert client.export() == self.CONFIRMATION

    def test_calls_correct_endpoint(self, client):
        client.session.get.return_value = make_response(200, self.CONFIRMATION)
        client.export()
        client.session.get.assert_called_once_with(
            f"{BASE_URL}/export", timeout=2.0
        )

    @pytest.mark.parametrize("status_code", [401, 500, 503])
    def test_raises_api_error_on_http_error(self, client, status_code):
        client.session.get.return_value = make_response(
            status_code, text="export failed: WAL is empty"
        )
        with pytest.raises(APIError):
            client.export()

    def test_raises_timeout_error_when_request_times_out(self, client):
        client.session.get.side_effect = RequestsTimeout()
        with pytest.raises(TimeoutError):
            client.export()

    def test_raises_connection_error_when_server_is_unreachable(self, client):
        client.session.get.side_effect = RequestsConnectionError()
        with pytest.raises(ConnectionError):
            client.export()
