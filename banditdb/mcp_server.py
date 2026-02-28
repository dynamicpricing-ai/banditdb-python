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
