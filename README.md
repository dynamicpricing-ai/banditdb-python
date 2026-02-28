# BanditDB Python SDK

The official Python client and Model Context Protocol (MCP) server for **BanditDB** — the ultra-fast, lock-free Contextual Bandit database written in Rust.

BanditDB abstracts away the complex linear algebra of Reinforcement Learning (LinUCB, Thompson Sampling) behind a dead-simple API. Build real-time personalizers, dynamic A/B tests, and give LLM agents mathematically rigorous persistent memory.

## Installation

```bash
pip install banditdb-python
```

Requires the BanditDB Rust server running (default: `http://localhost:8080`).

---

## 1. Standard SDK Usage

The client features automatic connection pooling, exponential backoff retries, and strict timeouts.

```python
from banditdb import Client, BanditDBError

# Connect to the BanditDB server.
# Pass api_key if BANDITDB_API_KEY is set on the server.
db = Client(
    url="http://localhost:8080",
    timeout=2.0,
    api_key="your-secret-key",   # omit if server runs without auth
)

try:
    # 1. Create a campaign (run once at startup)
    db.create_campaign(
        campaign_id="checkout_upsell",
        arms=["offer_discount", "offer_free_shipping"],
        feature_dim=3,
    )

    # 2. A user arrives — ask the database what to show them
    # Context: [is_mobile, cart_value_normalized, is_returning_user]
    arm_id, interaction_id = db.predict("checkout_upsell", [1.0, 0.8, 0.0])
    print(f"Showing: {arm_id}")  # e.g., "offer_free_shipping"

    # 3. The user clicked — send the reward
    db.reward(interaction_id, reward=1.0)

except BanditDBError as e:
    print(f"Database error: {e}")
```

### All Client methods

| Method | Description |
|--------|-------------|
| `health()` | Returns `True` if the server is reachable and healthy. |
| `create_campaign(campaign_id, arms, feature_dim)` | Register a new campaign. |
| `delete_campaign(campaign_id)` | Delete a campaign. Returns `False` if not found. |
| `predict(campaign_id, context)` | Returns `(arm_id, interaction_id)`. |
| `reward(interaction_id, reward)` | Close the feedback loop. Reward must be in `[0, 1]`. |
| `export()` | Trigger a Parquet export of the WAL. Returns the server confirmation message. |

---

## 2. The AI "Hive Mind" (Model Context Protocol)

Standard LLM agents are stateless — if they route a task to the wrong model and fail, they repeat the same mistake tomorrow. BanditDB's built-in MCP server gives the entire agent swarm shared persistent memory.

### Starting the MCP server

```bash
# Set environment variables before starting
export BANDITDB_URL=http://localhost:8080
export BANDITDB_API_KEY=your-secret-key   # omit if server runs without auth

banditdb-mcp
```

### Connecting to Claude Desktop

Add to your Claude configuration file:

- Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "banditdb": {
      "command": "banditdb-mcp",
      "args": [],
      "env": {
        "BANDITDB_URL": "http://localhost:8080",
        "BANDITDB_API_KEY": "your-secret-key"
      }
    }
  }
}
```

The agent swarm now has two tools: `get_intuition` and `record_outcome`. Every decision made by any agent in the network improves the routing for all future agents.

---

## 3. Data Science & Offline Evaluation

BanditDB event-sources every prediction and reward to a Write-Ahead Log (WAL). Export it to Apache Parquet for offline analysis with Polars or Pandas.

```python
# Trigger the export
message = db.export()
print(message)  # "Successfully exported N rows to bandit_logs_latest.parquet"

# Load into Polars for Offline Policy Evaluation
import polars as pl
df = pl.read_parquet("bandit_logs_latest.parquet")
predictions = df.select(pl.col("Predicted")).unnest("Predicted").drop_nulls()
print(predictions.head())
```

---

## Error Handling

| Exception | When raised |
|-----------|-------------|
| `BanditDBError` | Base exception — catch this to handle all SDK errors. |
| `ConnectionError` | Server is offline or unreachable. |
| `TimeoutError` | Request exceeded the configured timeout. |
| `APIError` | Server returned an error (e.g., campaign not found, unauthorized). |

---

## License

AGPL-3.0 — Copyright (C) 2026 Simeon Lukov.
See the [main repository](https://github.com/simeonlukov/banditdb) for details.
