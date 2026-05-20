import os
from mcp.server.fastmcp import FastMCP
from banditdb import Client, BanditDBError

# Initialize the MCP server
mcp = FastMCP("BanditDB Intuition Engine")

# Connect to the BanditDB server.
# Configure via environment variables:
#   BANDITDB_URL     — server address (default: http://localhost:8080)
#   BANDITDB_API_KEY — API key if the server requires authentication
db = Client(
    url=os.environ.get("BANDITDB_URL", "http://localhost:8080"),
    api_key=os.environ.get("BANDITDB_API_KEY"),
)

@mcp.tool()
def create_campaign(
    campaign_id: str,
    arms: list[str],
    feature_dim: int,
    alpha: float = 1.0,
    algorithm: str = "linucb",
) -> str:
    """
    Create a new decision campaign in BanditDB.

    Call this once before using get_intuition for a new type of decision.
    If the campaign already exists, this will return an error — that is safe
    to ignore if you are re-running setup code.

    Args:
        campaign_id: A unique name for this decision context (e.g., 'llm_routing',
                     'support_tier', 'offer_type'). Use snake_case.
        arms: The list of options the bandit will choose between
              (e.g., ['gpt-4o', 'claude-haiku', 'llama-3']).
        feature_dim: The number of floats in the context vector you will pass
                     to get_intuition. Must match exactly every time.
        alpha: Exploration coefficient (default 1.0). Lower values (e.g. 0.1)
               exploit learned knowledge faster. Higher values (e.g. 3.0) keep
               exploring uncertain arms longer.
        algorithm: Decision algorithm — "linucb" (default) or "thompson_sampling".
                   Use "thompson_sampling" for natural Bayesian exploration: no
                   alpha sweep needed, and concurrent users automatically diversify
                   arm coverage. Use "linucb" when you want deterministic,
                   predictable exploration you can tune via alpha.

    Returns:
        Confirmation that the campaign was created, or an error message.
    """
    try:
        db.create_campaign(campaign_id, arms, feature_dim, alpha=alpha, algorithm=algorithm)
        return (
            f"✅ Campaign '{campaign_id}' created with {len(arms)} arms: {arms}. "
            f"feature_dim={feature_dim}, alpha={alpha}, algorithm={algorithm}. "
            f"You can now call get_intuition('{campaign_id}', context) with a "
            f"context vector of {feature_dim} floats."
        )
    except BanditDBError as e:
        return f"Error creating campaign: {str(e)}"


@mcp.tool()
def list_campaigns() -> str:
    """
    List all active decision campaigns in BanditDB.

    Call this to discover what campaigns exist before calling get_intuition,
    or to check that a campaign was created successfully.

    Returns:
        A summary of all live campaigns with their arm count and alpha value,
        or a message if no campaigns exist yet.
    """
    try:
        campaigns = db.list_campaigns()
        if not campaigns:
            return "No campaigns found. Use create_campaign to create one."
        lines = [f"Active campaigns ({len(campaigns)}):"]
        for c in campaigns:
            lines.append(
                f"  • {c['campaign_id']} — {c['arm_count']} arms, "
                f"alpha={c['alpha']}, algorithm={c.get('algorithm', 'linucb')}"
            )
        return "\n".join(lines)
    except BanditDBError as e:
        return f"Error listing campaigns: {str(e)}"


@mcp.tool()
def campaign_diagnostics(campaign_id: str) -> str:
    """
    Inspect the learning state and exploration health of a campaign.

    Use this when you suspect a campaign is not learning, one arm is dominating,
    or rewards are not being received. Key signals:
    - entropy_status: "ok" means healthy. "warning"/"critical" means exploration
      has collapsed — one arm is absorbing most traffic without a convergence signal.
    - entropy_trend: "falling" means collapse happened recently (check for pipeline
      bugs or config changes). "stable" means the current state has been consistent.
    - likely_cause / suggested_action: present when entropy is degraded, tells you
      what likely happened and what to do about it.
    - theta_norm: 0.0 means this arm has never been rewarded. Growing means learning.
    - predictions vs rewards: a large gap means rewards are not closing the loop
      (TTL expiry, missing record_outcome calls, or a bug in your integration).

    Args:
        campaign_id: The campaign to inspect.

    Returns:
        A human-readable diagnostic report with per-arm statistics and entropy health.
    """
    try:
        d = db.diagnostics(campaign_id)
        entropy   = d.get("selection_entropy", "n/a")
        e_status  = d.get("entropy_status", "unknown")
        e_trend   = d.get("entropy_trend", "unknown")
        converged = d.get("converged")
        cause     = d.get("likely_cause")
        action    = d.get("suggested_action")

        lines = [
            f"Campaign: {d['campaign_id']} "
            f"(algorithm={d.get('algorithm', 'linucb')}, alpha={d.get('alpha', '?')})",
            f"Totals: {d['total_predictions']} predictions, {d['total_rewards']} rewards",
            "",
            f"Exploration health:",
            f"  entropy={entropy:.3f}  status={e_status}  trend={e_trend}"
            + (f"  converged={converged}" if converged is not None else ""),
        ]
        if cause:
            lines.append(f"  ⚠  likely_cause: {cause}")
        if action:
            lines.append(f"     suggested_action: {action}")

        lines += ["", "Arms:"]
        for arm_id, arm in sorted(d["arm_stats"].items()):
            reward_rate = (
                f"{arm['rewards'] / arm['predictions']:.0%}"
                if arm["predictions"] > 0
                else "n/a"
            )
            lines.append(
                f"  • {arm_id}: theta_norm={arm['theta_norm']:.4f}, "
                f"predictions={arm['predictions']}, "
                f"rewards={arm['rewards']} ({reward_rate})"
            )

        if d.get("challenger_traffic_pct") is not None:
            lines += [
                "",
                f"Tournament: challenger={d['challenger_traffic_pct']:.1f}%  "
                f"win_streak={d.get('tournament_win_streak', 0)}",
            ]
        return "\n".join(lines)
    except BanditDBError as e:
        return f"Error fetching campaign diagnostics: {str(e)}"


@mcp.tool()
def campaign_report(campaign_id: str) -> str:
    """
    Get the business-level convergence report for a campaign.

    Use this to answer: "Is this campaign done? Which arm is winning?"

    The `converged` field is the key signal:
      True  — the leading arm has a statistically significant advantage (95% CI).
              You can safely stop the experiment and deploy the winner.
      False — one arm leads but confidence intervals still overlap.
              Keep collecting data.
      None  — not enough data yet (< 30 rewards per arm).

    Also shows per-arm reward rates with confidence intervals, and for Progressive
    campaigns: challenger traffic percentage and win streak.

    Args:
        campaign_id: The campaign to evaluate.

    Returns:
        A human-readable convergence report with per-arm statistics.
    """
    try:
        r = db.report(campaign_id)
        lines = [
            f"Campaign: {r['campaign_id']}",
            f"Totals: {r['total_predictions']} predictions, {r['total_rewards']} rewards",
            f"Overall reward rate: {r['overall_reward_rate']:.1%}",
            f"Leading arm: {r.get('leading_arm', 'n/a')}",
            f"Converged: {r.get('converged')}",
            "",
            "Arms:",
        ]
        for arm_id, arm in r.get("arms", {}).items():
            ci_lo = arm.get("ci_lower", 0)
            ci_hi = arm.get("ci_upper", 0)
            lines.append(
                f"  • {arm_id}: reward_rate={arm.get('reward_rate', 0):.1%} "
                f"[{ci_lo:.3f}, {ci_hi:.3f}]  "
                f"predictions={arm.get('predictions', 0)}  "
                f"traffic={arm.get('traffic_share', 0):.1%}"
            )
        if r.get("challenger_traffic_pct") is not None:
            lines += [
                "",
                f"Tournament: challenger={r['challenger_traffic_pct']:.1f}%  "
                f"win_streak={r.get('tournament_win_streak', 0)}",
            ]
        return "\n".join(lines)
    except BanditDBError as e:
        return f"Error fetching campaign report: {str(e)}"


@mcp.tool()
def batch_get_intuition(predictions: list[dict]) -> str:
    """
    Ask BanditDB for the best action across multiple campaigns in a single round-trip.

    Use this instead of calling get_intuition repeatedly when you need decisions for
    multiple campaigns at once. More efficient — one network call instead of N.

    Each item in `predictions` must have:
        campaign_id : str   — the campaign to query
        context     : list  — feature vector matching that campaign's feature_dim

    Example input:
        [
          {"campaign_id": "llm_routing", "context": [0.8, 0.2, 0.5]},
          {"campaign_id": "support_tier", "context": [0.3, 0.9]}
        ]

    Args:
        predictions: List of {"campaign_id": str, "context": list[float]} dicts.

    Returns:
        One line per campaign with the suggested arm and interaction_id to save.
    """
    try:
        results = db.batch_predict(predictions)
        lines = [f"Batch results ({len(results)} campaigns):"]
        for i, r in enumerate(results):
            if "error" in r:
                campaign_id = predictions[i].get("campaign_id", f"item {i}")
                lines.append(f"  • {campaign_id}: ERROR — {r['error']}")
            else:
                lines.append(
                    f"  • {r.get('campaign_id', f'item {i}')}: "
                    f"arm='{r['arm_id']}'  interaction_id={r['interaction_id']}"
                )
        lines.append("")
        lines.append("[IMPORTANT] Save each interaction_id to call record_outcome later.")
        return "\n".join(lines)
    except BanditDBError as e:
        return f"Error in batch prediction: {str(e)}"


@mcp.tool()
def archive_campaign(campaign_id: str) -> str:
    """
    Soft-delete a campaign. The campaign stops accepting new predictions and rewards
    but all data (arm matrices, history) is preserved and can be restored.

    Use this instead of deleting when you want to pause a campaign without losing
    its learned weights — for example, to suspend a seasonal campaign and resume it
    next quarter with the accumulated training data intact.

    Args:
        campaign_id: The campaign to archive.

    Returns:
        Confirmation or error message.
    """
    try:
        db.archive_campaign(campaign_id)
        return (
            f"Campaign '{campaign_id}' archived. It will no longer accept predictions "
            f"or rewards. Use restore_campaign('{campaign_id}') to reactivate it."
        )
    except BanditDBError as e:
        return f"Error archiving campaign: {str(e)}"


@mcp.tool()
def restore_campaign(campaign_id: str) -> str:
    """
    Restore an archived campaign to active status.

    The campaign resumes accepting predictions and rewards with all previously
    learned weights intact — no retraining needed.

    Args:
        campaign_id: The archived campaign to restore.

    Returns:
        Confirmation or error message.
    """
    try:
        db.restore_campaign(campaign_id)
        return (
            f"Campaign '{campaign_id}' restored to active status. "
            f"It will resume learning from its previous state."
        )
    except BanditDBError as e:
        return f"Error restoring campaign: {str(e)}"


@mcp.tool()
def get_intuition(campaign_id: str, context: list[float]) -> str:
    """
    Ask the BanditDB Hive Mind for the best strategy or action to take.

    Args:
        campaign_id: The ID of the decision campaign (e.g., 'llm_routing', 'support_strategy').
        context: A list of floats representing the current state (e.g., user sentiment, task difficulty).

    Returns:
        A string telling you which action to take, and the interaction_id you MUST save for the reward.
    """
    try:
        arm_id, interaction_id = db.predict(campaign_id, context)
        return (
            f"💡 BanditDB Suggests: Take action '{arm_id}'.\n"
            f"[IMPORTANT] Save this interaction_id for the outcome: {interaction_id}"
        )
    except BanditDBError as e:
        return f"Error connecting to Hive Mind: {str(e)}"

@mcp.tool()
def record_outcome(interaction_id: str, reward: float) -> str:
    """
    Tell the BanditDB Hive Mind if the strategy it suggested was successful.

    Args:
        interaction_id: The unique ID returned by get_intuition.
        reward: 1.0 if the action was a massive success, 0.0 if it failed or was unhelpful.

    Returns:
        Confirmation that the global math matrices have been updated.
    """
    try:
        success = db.reward(interaction_id, reward)
        if success:
            return "🧠 Mathematical weights updated! The Swarm has learned from this interaction."
        return "Failed to update weights."
    except BanditDBError as e:
        return f"Error recording outcome: {str(e)}"

def main():
    """Entry point for the command-line interface."""
    #print("🚀 Starting BanditDB MCP Server...", flush=True)
    mcp.run()

if __name__ == "__main__":
    main()
