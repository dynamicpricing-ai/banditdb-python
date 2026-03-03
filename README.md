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
    # algorithm defaults to "linucb"; use "thompson_sampling" for Bayesian exploration
    db.create_campaign(
        campaign_id="checkout_upsell",
        arms=["offer_discount", "offer_free_shipping"],
        feature_dim=3,
    )
    # or: db.create_campaign(..., algorithm="thompson_sampling")

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
| `list_campaigns()` | Returns a list of all live campaigns with their `alpha` and `arm_count`. |
| `campaign_info(campaign_id)` | Returns the full diagnostic state for one campaign: per-arm `theta`, `theta_norm`, `prediction_count`, `reward_count`, and totals. Raises `APIError` (404) if not found. |
| `create_campaign(campaign_id, arms, feature_dim, alpha=1.0, algorithm="linucb")` | Register a new campaign. `algorithm` is `"linucb"` (default) or `"thompson_sampling"`. `alpha` controls exploration for both — for TS it sets the posterior width; `1.0` is the principled default. |
| `delete_campaign(campaign_id)` | Delete a campaign. Returns `False` if not found. |
| `predict(campaign_id, context)` | Returns `(arm_id, interaction_id)`. |
| `reward(interaction_id, reward)` | Close the feedback loop. Reward must be in `[0, 1]`. |
| `checkpoint()` | Flush the WAL, snapshot models, write Parquet files, rotate the WAL. Returns a summary string. |
| `export()` | List per-campaign Parquet files created by `checkpoint()`. Returns a formatted string. |

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

The agent swarm now has five tools:

| Tool | What it does |
|------|--------------|
| `create_campaign` | Create a new decision campaign. Accepts `algorithm` (`"linucb"` or `"thompson_sampling"`) and `alpha`. Use Thompson Sampling for natural Bayesian exploration with no tuning needed. |
| `list_campaigns` | List all active campaigns (shows `algorithm` and `alpha`) — useful to check what exists before calling `get_intuition`. |
| `campaign_diagnostics` | Inspect per-arm learning state: `theta_norm`, prediction counts, reward rates. Use this when a campaign doesn't seem to be learning. |
| `get_intuition` | Ask BanditDB which arm to pick for a given context. Returns the arm and an `interaction_id` to save. |
| `record_outcome` | Report whether the chosen action succeeded (1.0) or failed (0.0). Updates the shared model. |

Every decision made by any agent in the network improves the routing for all future agents.

---

## 3. Data Science & Offline Evaluation

BanditDB event-sources every prediction and reward to a Write-Ahead Log (WAL). Calling `checkpoint()` compiles completed prediction→reward pairs into Snappy-compressed Parquet files — one per campaign — for offline analysis with Polars or Pandas.

Every prediction is guaranteed to appear in the Parquet file even if its reward arrives hours later: BanditDB re-emits in-flight interactions at each checkpoint so delayed rewards are always captured in a future cycle.

```python
# Checkpoint: snapshot models, write Parquet, rotate the WAL.
# Call this on a schedule or after significant traffic.
summary = db.checkpoint()
print(summary)
# "Checkpoint written and WAL rotated: 2 campaigns, offset 4821 bytes,
#  150 interactions exported, 3 in-flight re-emitted"

# List which Parquet files are available
print(db.export())
# 'Parquet files in /data/exports: ["llm_routing.parquet"]'

# Load directly from the mounted volume into Polars for Offline Policy Evaluation.
# Flat schema: interaction_id | arm_id | reward | predicted_at | rewarded_at | feature_0 | ...
import polars as pl
df = pl.read_parquet("/data/exports/llm_routing.parquet")
print(df.head())
print(df.columns)
```

---

## Choosing an Algorithm

BanditDB supports two contextual bandit algorithms, selected at campaign creation time.

| Algorithm | `algorithm` value | Exploration style | When to use |
|-----------|------------------|-------------------|-------------|
| **LinUCB** | `"linucb"` (default) | Deterministic UCB bonus: `θ·x + α√(x·A⁻¹·x)` | Predictable, tunable. Sweep `alpha` offline to calibrate. |
| **Linear Thompson Sampling** | `"thompson_sampling"` | Samples θ̃ ~ N(θ, α²·A⁻¹), scores by θ̃·x | Bayesian posterior — no alpha-sweep needed. `alpha=1.0` is the natural posterior width. Concurrent users automatically diversify choices. |

```python
# LinUCB (default) — tune alpha to control how long it keeps exploring
db.create_campaign("routing", ["fast", "cheap"], feature_dim=4, alpha=1.5)

# Thompson Sampling — natural Bayesian exploration, alpha=1.0 is ideal
db.create_campaign("routing_ts", ["fast", "cheap"], feature_dim=4,
                   algorithm="thompson_sampling")
```

Both algorithms share identical state (A⁻¹, b, θ per arm), so the `predict` → `reward` loop is the same regardless of which you choose.

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
