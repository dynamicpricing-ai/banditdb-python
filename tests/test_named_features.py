"""
Coverage for named features: ``create_campaign(..., features=[...])`` and dict
contexts in ``predict`` / ``batch_predict`` / ``interact``.

The SDK reads the feature order from the campaign's metadata, appends the
constant 1 and normalises, so callers never assemble the vector by hand.
"""

import math

import pytest

from tests.conftest import BASE_URL, make_response

FEATURES = ["age", "income", "is_mobile"]


def _info(features=FEATURES, context={"bias": True, "normalize": True}, width=None):
    meta = {"features": features}
    if context is not None:
        meta["context"] = context
    width = width if width is not None else len(features) + 1
    return {
        "campaign_id": "c",
        "arms": {"a": {"theta": [0.0] * width}, "b": {"theta": [0.0] * width}},
        "metadata": meta,
    }


def _sent(client):
    return client.session.post.call_args.kwargs["json"]


def _unit(values):
    n = math.sqrt(sum(v * v for v in values))
    return [v / n for v in values]


class TestCreate:
    def test_features_set_dim_and_metadata(self, client):
        client.session.post.return_value = make_response(200, "Campaign Created")
        client.create_campaign("c", ["a", "b"], features=FEATURES, metadata={"owner": "x"})
        body = _sent(client)
        assert body["feature_dim"] == 4
        assert body["metadata"] == {
            "owner": "x",
            "features": FEATURES,
            "context": {"bias": True, "normalize": True},
        }

    def test_matching_feature_dim_is_accepted(self, client):
        client.session.post.return_value = make_response(200, "Campaign Created")
        client.create_campaign("c", ["a"], feature_dim=4, features=FEATURES)
        assert _sent(client)["feature_dim"] == 4

    def test_mismatched_feature_dim_rejected(self, client):
        with pytest.raises(ValueError, match="leave feature_dim out"):
            client.create_campaign("c", ["a"], feature_dim=3, features=FEATURES)

    def test_needs_features_or_dim(self, client):
        with pytest.raises(ValueError, match="features"):
            client.create_campaign("c", ["a"])

    @pytest.mark.parametrize("bad", [[], ["a", ""], ["a", "a"], ["a", 3]])
    def test_bad_feature_lists_rejected(self, client, bad):
        with pytest.raises(ValueError):
            client.create_campaign("c", ["a"], features=bad)
        client.session.post.assert_not_called()

    def test_conflicting_metadata_features_rejected(self, client):
        with pytest.raises(ValueError, match="disagrees"):
            client.create_campaign("c", ["a"], features=FEATURES, metadata={"features": ["x"]})

    def test_positional_feature_dim_still_works(self, client):
        client.session.post.return_value = make_response(200, "Campaign Created")
        client.create_campaign("c", ["a"], 5)
        assert _sent(client)["feature_dim"] == 5
        assert "metadata" not in _sent(client)

    def test_created_campaign_needs_no_lookup(self, client):
        client.session.post.side_effect = [
            make_response(200, "Campaign Created"),
            make_response(200, {"arm_id": "a", "interaction_id": "i"}),
        ]
        client.create_campaign("c", ["a"], features=FEATURES)
        client.predict("c", {"age": 0.5, "income": 0.2, "is_mobile": 1})
        client.session.get.assert_not_called()


class TestPredict:
    def test_dict_is_ordered_biased_and_normalised(self, client):
        client.session.get.return_value = make_response(200, _info())
        client.session.post.return_value = make_response(200, {"arm_id": "a", "interaction_id": "i"})
        arm, iid = client.predict("c", {"is_mobile": 1, "age": 0.5, "income": 0.2})
        assert (arm, iid) == ("a", "i")
        assert _sent(client)["context"] == pytest.approx(_unit([0.5, 0.2, 1.0, 1.0]))
        client.session.get.assert_called_once_with(f"{BASE_URL}/campaign/c", timeout=2.0)

    def test_layout_is_cached(self, client):
        client.session.get.return_value = make_response(200, _info())
        client.session.post.return_value = make_response(200, {"arm_id": "a", "interaction_id": "i"})
        for _ in range(3):
            client.predict("c", {"age": 0.5, "income": 0.2, "is_mobile": 1})
        assert client.session.get.call_count == 1

    def test_delete_forgets_layout(self, client):
        client.session.get.return_value = make_response(200, _info())
        client.session.post.return_value = make_response(200, {"arm_id": "a", "interaction_id": "i"})
        client.session.delete.return_value = make_response(200)
        client.predict("c", {"age": 0.5, "income": 0.2, "is_mobile": 1})
        client.delete_campaign("c")
        client.predict("c", {"age": 0.5, "income": 0.2, "is_mobile": 1})
        assert client.session.get.call_count == 2

    def test_missing_and_unknown_named(self, client):
        client.session.get.return_value = make_response(200, _info())
        with pytest.raises(ValueError, match=r"missing \['income'\].*unknown \['agee'\]"):
            client.predict("c", {"agee": 0.5, "age": 0.1, "is_mobile": 1})
        client.session.post.assert_not_called()

    @pytest.mark.parametrize("bad", ["0.5", None, True, float("nan")])
    def test_bad_value_named(self, client, bad):
        client.session.get.return_value = make_response(200, _info())
        with pytest.raises(ValueError, match="context\\['age'\\]"):
            client.predict("c", {"age": bad, "income": 0.2, "is_mobile": 1})

    def test_campaign_without_names(self, client):
        client.session.get.return_value = make_response(200, {"arms": {}, "metadata": None})
        with pytest.raises(ValueError, match="list of numbers"):
            client.predict("c", {"age": 0.5})

    def test_list_passes_through_without_lookup(self, client):
        client.session.post.return_value = make_response(200, {"arm_id": "a", "interaction_id": "i"})
        client.predict("c", [3.0, 4.0])
        assert _sent(client)["context"] == [3.0, 4.0]
        client.session.get.assert_not_called()

    def test_layout_flags_respected(self, client):
        client.session.get.return_value = make_response(
            200, _info(context={"bias": False, "normalize": False}, width=3)
        )
        client.session.post.return_value = make_response(200, {"arm_id": "a", "interaction_id": "i"})
        client.predict("c", {"age": 0.5, "income": 0.2, "is_mobile": 1})
        assert _sent(client)["context"] == [0.5, 0.2, 1.0]


class TestLegacyConsoleCampaigns:
    """Console campaigns stored ``features`` before ``context`` existed."""

    def test_width_with_constant(self, client):
        client.session.get.return_value = make_response(200, _info(context=None))
        client.session.post.return_value = make_response(200, {"arm_id": "a", "interaction_id": "i"})
        client.predict("c", {"age": 0.5, "income": 0.2, "is_mobile": 1})
        assert len(_sent(client)["context"]) == 4

    def test_width_without_constant(self, client):
        client.session.get.return_value = make_response(200, _info(context=None, width=3))
        client.session.post.return_value = make_response(200, {"arm_id": "a", "interaction_id": "i"})
        client.predict("c", {"age": 0.5, "income": 0.2, "is_mobile": 1})
        assert _sent(client)["context"] == pytest.approx(_unit([0.5, 0.2, 1.0]))


class TestOtherCalls:
    def test_batch_predict_resolves_each_item(self, client):
        client.session.get.return_value = make_response(200, _info())
        client.session.post.return_value = make_response(200, [])
        client.batch_predict([
            {"campaign_id": "c", "context": {"age": 0.5, "income": 0.2, "is_mobile": 1}},
            {"campaign_id": "d", "context": [0.1, 0.2], "exclude_arms": ["x"]},
        ])
        items = _sent(client)["predictions"]
        assert len(items[0]["context"]) == 4
        assert items[1] == {"campaign_id": "d", "context": [0.1, 0.2], "exclude_arms": ["x"]}

    def test_batch_error_names_item(self, client):
        client.session.get.return_value = make_response(200, _info())
        with pytest.raises(ValueError, match=r"predictions\[0\]: .*missing"):
            client.batch_predict([{"campaign_id": "c", "context": {"age": 0.5}}])

    def test_batch_does_not_mutate_input(self, client):
        client.session.get.return_value = make_response(200, _info())
        client.session.post.return_value = make_response(200, [])
        item = {"campaign_id": "c", "context": {"age": 0.5, "income": 0.2, "is_mobile": 1}}
        client.batch_predict([item])
        assert isinstance(item["context"], dict)

    def test_interact_accepts_dict(self, client):
        client.session.get.return_value = make_response(200, _info())
        client.session.post.return_value = make_response(200, {"interaction_id": "i"})
        client.interact("c", "a", {"age": 0.5, "income": 0.2, "is_mobile": 1}, 1.0)
        assert len(_sent(client)["context"]) == 4
