# Changelog

## 0.2.0 — BanditDB server 2.0.0 compatibility

**Requires server 2.0.0+.** Against older servers, `health_detail()` raises a clear
`APIError` explaining the mismatch; everything else still works.

### Fixed

- **`health_detail()` was broken against server 2.0.0.** It called `GET /health`,
  which no longer returns campaign data — the campaign map moved to
  `/health/detail` because those identifiers carry the tenant prefix and `/health`
  is unauthenticated. Every caller of `health_detail()["campaigns"]` was hitting
  `KeyError`. Now targets the correct endpoint and requires a reader key.

### Added

- **`normalize_context(context)`** — scale a feature vector to unit L2 norm.
  Worth using: LinUCB's exploration term scales with `‖x‖` and its regret analysis
  assumes `‖x‖ ≤ 1`. Unnormalised input still converges, just far more slowly and
  with no error to tell you. On the UCI shuttle benchmark this alone moved
  cumulative regret from 2,026 to 709.
- **`server_info()`** — returns `{status, version, features}`. Use `features` to
  check whether the server can run neural algorithms before creating such a
  campaign. Returns `{}` for pre-2.0.0 servers.
- **`interact(campaign_id, arm_id, context, reward)`** — record a decision made
  elsewhere along with its outcome, for backfilling from historical logs. The
  endpoint existed on the server but the SDK never exposed it.
- **Client-side validation** on `predict`, `batch_predict`, `reward`, and
  `interact`. The server rejects these with 400; checking locally raises
  `ValueError` naming the offending index and skips the round trip:
  - contexts must be non-empty, finite, and within ±1e6 (`MAX_CONTEXT_MAGNITUDE`)
  - rewards must be finite and within `[0.0, 1.0]`

  The magnitude bound is not paranoia. A value near 1e155 is finite but squares to
  infinity inside the server's rank-one update, and the resulting NaN persists
  through the checkpoint and survives restart — the campaign is permanently dead.

### Changed

- **`reward()` now waits for the server to fsync.** A success response means the
  reward is on disk and survives power loss, which costs roughly 3.4 ms. The server
  amortises the fsync across concurrent callers, so parallel submission reaches
  ~4,400 rewards/s while a serial loop pays the full latency every call. Batch or
  thread if throughput matters.
- MCP tools now catch `ValueError` and return a readable message instead of raising
  an unhandled exception at the caller.
- `record_outcome` and `get_intuition` tool descriptions state the accepted ranges,
  since the model reads them and would otherwise pass out-of-range values.

## 0.1.6 and earlier

See the git history.
