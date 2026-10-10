"""
Coverage for budget constraints (Lagrangian pacing), added in BanditDB server 2.3.0.

The server takes a ``pacing`` block on campaign creation, ``costs`` when adding an
arm, and reports budget state at ``GET /campaign/:id/pacing``. The client mirrors the
server's validation locally, and additionally rejects ``arm_costs`` keys that are not
campaign arms — the server accepts those silently, leaving the real arm free.
"""

import math

import pytest

from banditdb import PacingConfig, ResourceConstraint
from banditdb.exceptions import APIError
from tests.conftest import BASE_URL, make_response


def _pacing(**overrides):
    fields = dict(name="promo", budget=100.0, arm_costs={"b": 1.0})
    fields.update(overrides)
    return PacingConfig(resources=[ResourceConstraint(**fields)])


class TestSerialisation:
    def test_defaults_omit_optional_tuning(self):
        assert _pacing()._to_api() == {
            "resources": [
                {"name": "promo", "budget": 100.0, "horizon": 10_000, "arm_costs": {"b": 1.0}}
            ],
            "adaptive": True,
        }

    def test_tuning_fields_sent_when_set(self):
        r = _pacing(step_size=0.01, lambda_max=5.0, initial_lambda=0.0)._to_api()["resources"][0]
        assert r["step_size"] == 0.01
        assert r["lambda_max"] == 5.0
        assert r["initial_lambda"] == 0.0


class TestValidation:
    @pytest.mark.parametrize("budget", [0, -1.0, math.inf, math.nan])
    def test_budget_must_be_positive_and_finite(self, budget):
        with pytest.raises(ValueError, match="budget"):
            _pacing(budget=budget)._validate()

    @pytest.mark.parametrize("horizon", [0, -5, 1.5, True])
    def test_horizon_must_be_positive_int(self, horizon):
        with pytest.raises(ValueError, match="horizon"):
            _pacing(horizon=horizon)._validate()

    @pytest.mark.parametrize("field", ["step_size", "lambda_max"])
    def test_tuning_must_be_positive(self, field):
        with pytest.raises(ValueError, match=field):
            _pacing(**{field: 0.0})._validate()

    def test_initial_lambda_may_be_zero_not_negative(self):
        _pacing(initial_lambda=0.0)._validate()
        with pytest.raises(ValueError, match="initial_lambda"):
            _pacing(initial_lambda=-0.1)._validate()

    def test_costs_must_be_non_negative(self):
        _pacing(arm_costs={"b": 0.0})._validate()
        with pytest.raises(ValueError, match=r"arm_costs\['b'\]"):
            _pacing(arm_costs={"b": -1.0})._validate()

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            _pacing(name="  ")._validate()

    def test_duplicate_names_rejected(self):
        cfg = PacingConfig(resources=[
            ResourceConstraint(name="x", budget=1.0),
            ResourceConstraint(name="x", budget=2.0),
        ])
        with pytest.raises(ValueError, match="duplicate"):
            cfg._validate()


class TestCreateCampaign:
    def test_sends_pacing(self, client):
        client.session.post.return_value = make_response(200, "Campaign Created")
        assert client.create_campaign("c", ["a", "b"], 2, pacing=_pacing())
        body = client.session.post.call_args.kwargs["json"]
        assert body["pacing"]["resources"][0]["arm_costs"] == {"b": 1.0}

    def test_omits_pacing_by_default(self, client):
        client.session.post.return_value = make_response(200, "Campaign Created")
        client.create_campaign("c", ["a", "b"], 2)
        assert "pacing" not in client.session.post.call_args.kwargs["json"]

    def test_unknown_arm_in_costs_rejected_before_sending(self, client):
        with pytest.raises(ValueError, match=r"unknown arms \['bb'\]"):
            client.create_campaign("c", ["a", "b"], 2, pacing=_pacing(arm_costs={"bb": 1.0}))
        client.session.post.assert_not_called()

    def test_invalid_pacing_rejected_before_sending(self, client):
        with pytest.raises(ValueError):
            client.create_campaign("c", ["a", "b"], 2, pacing=_pacing(budget=0))
        client.session.post.assert_not_called()


class TestAddArm:
    def test_minimal_body(self, client):
        client.session.post.return_value = make_response(200, "Arm Added")
        assert client.add_arm("c", "new") is True
        args, kwargs = client.session.post.call_args
        assert args[0] == f"{BASE_URL}/campaign/c/arms"
        assert kwargs["json"] == {"arm_id": "new"}

    def test_full_body(self, client):
        client.session.post.return_value = make_response(200, "Arm Added")
        client.add_arm(
            "c", "new", group="g",
            warm_start={"from": "group", "strength": 2.0},
            costs={"promo": 3.0},
        )
        assert client.session.post.call_args.kwargs["json"] == {
            "arm_id": "new",
            "group": "g",
            "warm_start": {"from": "group", "strength": 2.0},
            "costs": {"promo": 3.0},
        }

    def test_negative_cost_rejected_before_sending(self, client):
        with pytest.raises(ValueError, match="costs"):
            client.add_arm("c", "new", costs={"promo": -1.0})
        client.session.post.assert_not_called()

    def test_server_error_surfaces(self, client):
        client.session.post.return_value = make_response(400, text="unknown resource 'x'")
        with pytest.raises(APIError, match="unknown resource"):
            client.add_arm("c", "new", costs={"x": 1.0})


class TestPacingReport:
    def test_returns_report(self, client):
        report = {
            "adaptive": True,
            "resources": [{
                "name": "promo", "budget": 100.0, "consumed": 25.0, "remaining": 75.0,
                "utilization": 0.25, "lambda": 0.1, "horizon": 1000, "decisions": 250,
                "is_exhausted": False,
            }],
        }
        client.session.get.return_value = make_response(200, report)
        assert client.pacing("c") == report
        assert client.session.get.call_args.args[0] == f"{BASE_URL}/campaign/c/pacing"

    def test_none_for_unconstrained_campaign(self, client):
        client.session.get.return_value = make_response(200, None, text="null")
        assert client.pacing("c") is None
