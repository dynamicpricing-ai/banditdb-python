# BanditDB Python SDK 🚀

The official Python client and Model Context Protocol (MCP) server for **BanditDB** — the ultra-fast, lock-free Contextual Bandit database written in Rust.

BanditDB abstracts away the complex linear algebra of Reinforcement Learning (LinUCB, Thompson Sampling) behind a dead-simple API. It allows developers to build real-time personalizers, dynamic A/B tests, and gives LLM Agents mathematically rigorous "persistent memory."

## 📦 Installation

```bash
pip install banditdb

Note: You must have the BanditDB Rust Server running (default: http://localhost:8080).

💻 1. Standard SDK Usage (For Developers & Data Scientists)
The Python client is production-ready, featuring automatic connection pooling, exponential backoff retries, and strict timeouts.

Quickstart

```python
from banditdb import Client, BanditDBError

# Connect to the BanditDB Rust backend
db = Client(url="http://localhost:8080", timeout=2.0)

try:
    # 1. Create a dynamic campaign (Cold Start)
    # E.g., 3 features in our context vector, 2 possible actions
    db.create_campaign(
        campaign_id="checkout_upsell", 
        arms=["offer_discount", "offer_free_shipping"], 
        feature_dim=3
    )

    # 2. A user arrives! Ask the database what to show them.
    # Context vector could be:[is_mobile, cart_value_normalized, is_returning_user]
    user_context = [1.0, 0.8, 0.0]
    
    arm_id, interaction_id = db.predict("checkout_upsell", user_context)
    print(f"Showing user: {arm_id}") # e.g., "offer_free_shipping"

    # ... User interacts with your app ...

    # 3. The user clicked the offer! Send a reward of 1.0.
    # The Sherman-Morrison matrices in Rust are instantly updated.
    db.reward(interaction_id, reward=1.0)
    print("Database mathematically updated.")

except BanditDBError as e:
    print(f"Database error: {e}")
```


🤖 2. The AI "Hive Mind" (Model Context Protocol)

Standard LLMs (like Claude or GPT-4) are amnesiacs. If they try a tool and fail, they might try the exact same failing strategy in a new chat tomorrow.
BanditDB includes a built-in MCP Server that acts as a shared mathematical "Intuition Engine" for AI agents.
Starting the MCP Server
Because you installed the banditdb package, you automatically have the MCP server installed globally as a command-line tool. You can test it by running:

```bash
banditdb-mcp

Connecting Claude Desktop to BanditDB
To give Claude persistent reinforcement-learning memory across all your chats, add BanditDB to your Claude configuration file:

Mac: ~/Library/Application Support/Claude/claude_desktop_config.json
Windows: %APPDATA%\Claude\claude_desktop_config.json

```JSON
{
  "mcpServers": {
    "banditdb": {
      "command": "banditdb-mcp",
      "args":[]
    }
  }
}
```

Restart Claude. You can now prompt Claude with:

"I need to parse a massive JSON file. Can you ask BanditDB whether I should use jq or python? My context vector is [file_size_gb, has_nested_arrays]."

Claude will consult the database, take the action, and record the outcome (reward) automatically, optimizing itself for the next time you ask!

📊 3. Data Science & Offline Evaluation
BanditDB uses Event Sourcing. Every prediction and reward is asynchronously written to a Write-Ahead Log (WAL). You can export this log into heavily compressed Apache Parquet format via the Rust API for offline model training using Polars or Pandas.


```python
import polars as pl
import requests

# Trigger the Rust backend to compile the Parquet data lake
requests.get("http://localhost:8080/export")

# Load the exact historical DNA of your database
df = pl.read_parquet("bandit_logs_latest.parquet")

# View all predictions and propensity scores for Offline Policy Evaluation (OPE)
predictions = df.select(pl.col("Predicted")).unnest("Predicted").drop_nulls()
print(predictions.head())
```

🛠️ Error Handling
The SDK exposes specific exceptions for robust error handling in production:
- BanditDBError: Base exception.
- ConnectionError: The Rust server is offline or unreachable.
- TimeoutError: The database took too long to respond (protects your app from hanging).
- APIError: The database returned an error (e.g., Campaign Not Found).

License
MIT License.

