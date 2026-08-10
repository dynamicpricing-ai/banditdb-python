"""
Coverage for the client changes that track BanditDB server 2.0.0.

The server tightened what it accepts and moved one endpoint. These tests pin the
client's side of that:

  * ``health_detail()`` targets ``/health/detail`` — the public ``/health`` no longer
    carries campaign data, so the old target raised KeyError against any 2.0.0 server.
  * Contexts and rewards are checked locally. The server rejects the same values with
    400; catching them here names the offending index and saves a round trip.
  * ``normalize_context()`` exists because scaling matters more than it looks: on the
    UCI shuttle benchmark, unit-L2 input moved cumulative regret from 2,026 to 709.
"""

import math

import pytest

from banditdb import MAX_CONTEXT_MAGNITUDE, normalize_context
from banditdb.client import _validate_context, _validate_reward


class TestNormalizeContext:
    def test_scales_to_unit_norm(self):
        out = normalize_context([3.0, 4.0])
        assert out == pytest.approx([0.6, 0.8])
        assert math.sqrt(sum(v * v for v in out)) == pytest.approx(1.0)

    def test_preserves_direction(self):
        out = normalize_context([2.0, 0.0, 0.0])
        assert out == pytest.approx([1.0, 0.0, 0.0])

    def test_all_zero_vector_is_returned_unchanged(self):
        # Dividing by a zero norm would produce NaN, which is exactly what the
        # server rejects — so this must not "normalise" into invalid input.
        assert normalize_context([0.0, 0.0]) == [0.0, 0.0]

    def test_does_not_mutate_input(self):
        original = [3.0, 4.0]
        normalize_context(original)
        assert original == [3.0, 4.0]

    def test_large_values_become_acceptable(self):
        # The point of the helper: input the server would reject becomes valid.
        huge = [1e9, 2e9]
        with pytest.raises(ValueError):
            _validate_context(huge)
        _validate_context(normalize_context(huge))  # must not raise


class TestContextValidation:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite(self, bad):
        with pytest.raises(ValueError, match="NaN and infinity"):
            _validate_context([0.1, bad])

    @pytest.mark.parametrize("bad", [1e200, -1e200, 1e155])
    def test_rejects_magnitudes_that_overflow_the_update(self, bad):
        with pytest.raises(ValueError, match="magnitude"):
            _validate_context([bad])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_context([])

    def test_names_the_offending_index(self):
        with pytest.raises(ValueError, match=r"context\[2\]"):
            _validate_context([0.1, 0.2, float("nan")])

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="must be a number"):
            _validate_context([0.1, "0.2"])

    def test_accepts_values_at_the_limit(self):
        _validate_context([MAX_CONTEXT_MAGNITUDE, -MAX_CONTEXT_MAGNITUDE])

    def test_accepts_ordinary_input(self):
        _validate_context([0.0, -1.0, 0.5, 42.0])


class TestRewardValidation:
    @pytest.mark.parametrize("bad", [1.5, -0.1, 5.0, 100.0])
    def test_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError, match=r"outside \[0.0, 1.0\]"):
            _validate_reward(bad)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_non_finite(self, bad):
        with pytest.raises(ValueError, match="finite"):
            _validate_reward(bad)

    @pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
    def test_accepts_the_documented_range(self, good):
        _validate_reward(good)

    def test_message_suggests_rescaling(self):
        # An out-of-range reward is almost always an unscaled business metric.
        with pytest.raises(ValueError, match="Rescale"):
            _validate_reward(42.0)


class TestClientGuardsBeforeSending:
    """Validation must run before any HTTP call, not after a 400 comes back."""

    def test_predict_rejects_without_calling_server(self, client):
        with pytest.raises(ValueError):
            client.predict("c", [float("nan")])
        client.session.post.assert_not_called()

    def test_reward_rejects_without_calling_server(self, client):
        with pytest.raises(ValueError):
            client.reward("iid", 5.0)
        client.session.post.assert_not_called()

    def test_batch_predict_checks_every_item_first(self, client):
        items = [
            {"campaign_id": "c", "context": [0.1]},
            {"campaign_id": "c", "context": [float("inf")]},
        ]
        with pytest.raises(ValueError, match=r"predictions\[1\]"):
            client.batch_predict(items)
        client.session.post.assert_not_called()

    def test_batch_predict_requires_both_keys(self, client):
        with pytest.raises(ValueError, match="campaign_id"):
            client.batch_predict([{"context": [0.1]}])
        client.session.post.assert_not_called()

    def test_interact_validates_both_arguments(self, client):
        with pytest.raises(ValueError):
            client.interact("c", "A", [0.1], 9.0)
        with pytest.raises(ValueError):
            client.interact("c", "A", [float("nan")], 1.0)
        client.session.post.assert_not_called()
