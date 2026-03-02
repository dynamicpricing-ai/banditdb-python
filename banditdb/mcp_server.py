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
               exploring uncertain arms longer. Use the default unless you have
               a specific reason to change it.

    Returns:
        Confirmation that the campaign was created, or an error message.
    """
    try:
        db.create_campaign(campaign_id, arms, feature_dim, alpha=alpha)
        return (
            f"✅ Campaign '{campaign_id}' created with {len(arms)} arms: {arms}. "
            f"feature_dim={feature_dim}, alpha={alpha}. "
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
                f"  • {c['campaign_id']} — {c['arm_count']} arms, alpha={c['alpha']}"
            )
        return "\n".join(lines)
    except BanditDBError as e:
        return f"Error listing campaigns: {str(e)}"


@mcp.tool()
def campaign_diagnostics(campaign_id: str) -> str:
    """
    Inspect the learning state of a campaign to diagnose whether it is working.

    Use this when you suspect a campaign is not learning, one arm is dominating,
    or rewards are not being received. The key signals are:
    - theta_norm: 0.0 means this arm has never been rewarded. Growing means learning.
    - prediction_count vs reward_count: a large gap means rewards are not closing
      the loop (TTL expiry, missing record_outcome calls, or a bug in your code).

    Args:
        campaign_id: The campaign to inspect.

    Returns:
        A human-readable diagnostic report with per-arm statistics.
    """
    try:
        info = db.campaign_info(campaign_id)
        lines = [
            f"Campaign: {info['campaign_id']} (alpha={info['alpha']})",
            f"Totals: {info['total_predictions']} predictions, "
            f"{info['total_rewards']} rewards",
            "",
            "Arms:",
        ]
        for arm_id, arm in sorted(info["arms"].items()):
            reward_rate = (
                f"{arm['reward_count'] / arm['prediction_count']:.0%}"
                if arm["prediction_count"] > 0
                else "n/a"
            )
            lines.append(
                f"  • {arm_id}: theta_norm={arm['theta_norm']:.4f}, "
                f"predictions={arm['prediction_count']}, "
                f"rewards={arm['reward_count']} ({reward_rate})"
            )
        return "\n".join(lines)
    except BanditDBError as e:
        return f"Error fetching campaign diagnostics: {str(e)}"


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
    print("🚀 Starting BanditDB MCP Server...", flush=True)
    mcp.run()

if __name__ == "__main__":
    main()
