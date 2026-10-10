"""
Coverage for arm lifecycle: persistent status (``set_arm_status``) and per-request
candidate filters (``eligible_arms`` / ``exclude_arms`` on predict).
"""

import pytest

from tests.conftest import BASE_URL, make_response


class TestSetArmStatus:
    @pytest.mark.parametrize("status", ["active", "paused", "retired"])
    def test_sends_status(self, client, status):
        client.session.post.return_value = make_response(200, "Arm Status Updated")
        assert client.set_arm_status("c", "a1", status) is True
        args, kwargs = client.session.post.call_args
        assert args[0] == f"{BASE_URL}/campaign/c/arms/a1/status"
        assert kwargs["json"] == {"status": status}

    def test_unknown_status_rejected_before_sending(self, client):
        with pytest.raises(ValueError, match="status must be one of"):
            client.set_arm_status("c", "a1", "deleted")
        client.session.post.assert_not_called()


class TestPredictFilters:
    def test_filters_omitted_by_default(self, client):
        client.session.post.return_value = make_response(
            200, {"arm_id": "a", "interaction_id": "i"}
        )
        client.predict("c", [0.1])
        assert client.session.post.call_args.kwargs["json"] == {
            "campaign_id": "c", "context": [0.1],
        }

    def test_filters_sent(self, client):
        client.session.post.return_value = make_response(
            200, {"arm_id": "a", "interaction_id": "i"}
        )
        client.predict("c", [0.1], eligible_arms=["a", "b"], exclude_arms=["b"])
        body = client.session.post.call_args.kwargs["json"]
        assert body["eligible_arms"] == ["a", "b"]
        assert body["exclude_arms"] == ["b"]


class TestMcpTools:
    @pytest.fixture
    def mcp(self, monkeypatch, client):
        from banditdb import mcp_server
        monkeypatch.setattr(mcp_server, "db", client)
        return mcp_server

    def test_create_campaign_with_budgets(self, mcp, client):
        client.session.post.return_value = make_response(200, "Campaign Created")
        out = mcp.create_campaign(
            "c", ["a", "b"], 2,
            budgets=[{"name": "promo", "budget": 10, "arm_costs": {"b": 1.0}}],
        )
        assert "budgets=['promo']" in out
        pacing = client.session.post.call_args.kwargs["json"]["pacing"]
        assert pacing["resources"][0]["horizon"] == 10_000

    def test_create_campaign_budget_missing_field(self, mcp):
        assert "missing" in mcp.create_campaign("c", ["a"], 2, budgets=[{"name": "x"}])

    def test_campaign_budget_without_pacing(self, mcp, client):
        client.session.get.return_value = make_response(200, None, text="null")
        assert "no budget constraints" in mcp.campaign_budget("c")

    def test_campaign_budget_exhausted(self, mcp, client):
        client.session.get.return_value = make_response(200, {
            "adaptive": True,
            "resources": [{
                "name": "promo", "budget": 10.0, "consumed": 10.0, "remaining": 0.0,
                "utilization": 1.0, "lambda": 1.0, "horizon": 50, "decisions": 60,
                "is_exhausted": True,
            }],
        })
        out = mcp.campaign_budget("c")
        assert "promo: 10 / 10 spent (100%)" in out
        assert "EXHAUSTED" in out

    def test_add_arm_warm_start(self, mcp, client):
        client.session.post.return_value = make_response(200, "Arm Added")
        mcp.add_arm("c", "new", warm_start_from="population", costs={"promo": 2.0})
        assert client.session.post.call_args.kwargs["json"] == {
            "arm_id": "new",
            "warm_start": {"from": "population"},
            "costs": {"promo": 2.0},
        }

    def test_set_arm_status_invalid(self, mcp):
        assert mcp.set_arm_status("c", "a", "gone").startswith("Invalid input")
