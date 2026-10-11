# Changelog

## 0.3.1 — NeuralTS

### Added

- **`NeuralTSConfig`** — NeuralTS, the server's `neural_thompson_sampling`
  algorithm (Zhang et al., ICLR 2021) had no SDK support at all. Same parameters as
  `NeuralLinUCBConfig`, but posterior-sampling exploration scaled by `alpha`.
  Usable directly or as a Progressive `base` / `challenger`.

### Fixed

- The terminal dashboard coloured `thompson_sampling` campaigns as unknown.

## 0.3.0 — Budget constraints and arm lifecycle (server 2.3.0)

### Added

- **Budget constraints (Lagrangian pacing).** `create_campaign(..., pacing=PacingConfig(...))`
  attaches one or more `ResourceConstraint`s — `name`, `budget`, `horizon`, per-arm
  `arm_costs`, and optional `step_size` / `lambda_max` / `initial_lambda` tuning.
  Requires server 2.3.0+; older servers ignore the field and the campaign runs
  unconstrained.
- **`pacing(campaign_id)`** — budget, consumption, remaining, utilization, shadow
  price and exhaustion per resource. `None` for campaigns without pacing.
  `campaign_info()` also carries this under `pacing`.
- **`add_arm(campaign_id, arm_id, group=None, warm_start=None, costs=None)`** — the
  server's add-arm endpoint was never exposed by the SDK. `costs` sets the new arm's
  per-resource cost on a pacing campaign.
- Local validation mirrors the server (positive finite budgets, positive horizon,
  non-negative costs, unique resource names) and goes one further: `arm_costs` keys
  that are not campaign arms raise `ValueError`. The server accepts them silently,
  leaving the misspelt arm free.
- **`set_arm_status(campaign_id, arm_id, status)`** — pause, retire, or reactivate
  an arm (`"active"` / `"paused"` / `"retired"`). Exported `ArmStatus` type.
- **`predict(..., eligible_arms=None, exclude_arms=None)`** — per-request candidate
  filters, for constraints that change per call (out of stock for this shopper,
  already shown). `batch_predict` items accept the same keys.
- **MCP server: three new tools and two extended ones.** `campaign_budget`,
  `add_arm`, `set_arm_status` are new; `create_campaign` takes `budgets` and
  `get_intuition` takes `exclude_arms` / `eligible_arms`. Twelve tools in total.

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

### Packaging — three fixes found while building 0.2.0

- **`import banditdb` failed on a clean install.** `__init__.py` eagerly imported
  `eval`, which imports numpy — but numpy lives under the optional `[eval]` extra.
  A plain `pip install banditdb-python` produced an unusable package unless numpy
  happened to be present for other reasons. **This affects 0.1.6 on PyPI today.**
  The submodule now loads on first access, and a missing extra raises an actionable
  message instead of `ModuleNotFoundError`.
- **`mcp` was unbounded (`>=1.0.0`) and resolves to 2.0.0**, which removed
  `mcp.server.fastmcp`. Any install today gets an MCP server that cannot import.
  Pinned to `>=1.0.0,<2`, matching the API the code targets. **Also affects 0.1.6.**
- **`banditdb/.claude/settings.local.json` was packaged into the wheel and sdist**,
  and it contained an API key. Never committed to git and the key is already dead,
  but it would have gone to PyPI permanently. Build now excludes dotfiles, caches,
  and stale `*.egg-info`.

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
