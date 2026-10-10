import os
from mcp.server.fastmcp import FastMCP
from banditdb import Client, BanditDBError, PacingConfig, ResourceConstraint

# Initialize the MCP server
# FastMCP takes no `version` argument in mcp 1.x, so serverInfo reports the
# library's version rather than ours. Cosmetic, and not worth breaking the server
# over. `website_url` is supported and is what directories link to.
mcp = FastMCP(
    "BanditDB Intuition Engine",
    website_url="https://banditdb.com/docs/mcp/",
)

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
    decay_half_life_hours: float | None = None,
    budgets: list[dict] | None = None,
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
        decay_half_life_hours: Exponential forgetting. Confidence halves every N hours
                   of real time, applied at each checkpoint. The learned direction
                   (theta) is preserved — only certainty erodes, so the bandit
                   re-explores stale knowledge without forgetting what it learned.
                   None = no forgetting (default). Recommended values:
                   24.0 for fast-changing contexts (daily drift),
                   168.0 for weekly drift (good default for agentic memory),
                   720.0 for slow monthly drift.
        budgets: Optional budget constraints, for when some actions cost something
                   limited (discount spend, human-agent minutes, paid API calls).
                   One dict per resource:
                     {"name": "promo_budget", "budget": 5000, "horizon": 100000,
                      "arm_costs": {"discount_10": 1.0, "discount_20": 2.0}}
                   `budget` is the total available over `horizon` decisions
                   (default 10000). `arm_costs` is what each arm consumes per
                   suggestion; arms left out are free and every key must be one of
                   `arms`. BanditDB then spreads spend over the horizon, picking
                   costly arms only where they are worth it. Budget is spent when
                   get_intuition suggests an arm, never resets, and cannot be changed
                   after creation. Requires BanditDB server 2.3.0+.

    Returns:
        Confirmation that the campaign was created, or an error message.
    """
    try:
        pacing = None
        if budgets:
            pacing = PacingConfig(resources=[
                ResourceConstraint(
                    name=b["name"],
                    budget=b["budget"],
                    horizon=b.get("horizon", 10_000),
                    arm_costs=b.get("arm_costs", {}),
                )
                for b in budgets
            ])
        db.create_campaign(
            campaign_id, arms, feature_dim,
            alpha=alpha, algorithm=algorithm,
            decay_half_life_hours=decay_half_life_hours,
            pacing=pacing,
        )
        decay_note = (
            f", decay_half_life={decay_half_life_hours}h"
            if decay_half_life_hours is not None else ""
        )
        if pacing is not None:
            decay_note += f", budgets={[r.name for r in pacing.resources]}"
        return (
            f"✅ Campaign '{campaign_id}' created with {len(arms)} arms: {arms}. "
            f"feature_dim={feature_dim}, alpha={alpha}, algorithm={algorithm}{decay_note}. "
            f"You can now call get_intuition('{campaign_id}', context) with a "
            f"context vector of {feature_dim} floats."
        )
    except KeyError as e:
        return f"Invalid input: each budget needs 'name' and 'budget' (missing {e})"
    except ValueError as e:
        return f"Invalid input: {str(e)}"
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
    except ValueError as e:
        return f"Invalid input: {str(e)}"
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
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except BanditDBError as e:
        return f"Error fetching campaign diagnostics: {str(e)}"


@mcp.tool()
def campaign_report(campaign_id: str) -> str:
    """
    Get per-arm performance for a campaign, with confidence intervals.

    Use this to answer: "How is this campaign doing? Which arm leads, and by how much?"

    The `converged` field tests whether ONE arm is best for EVERYBODY:
      True  — the leading arm has a statistically significant global advantage
              (95% CI). Safe to stop and deploy the winner.
      False — no single global winner yet.
      None  — not enough data yet (< 30 rewards per arm).

    IMPORTANT: on a contextual campaign, `converged` stays False by design. Where
    different arms win for different contexts — one action for mobile users,
    another for desktop — no arm is globally best, so the intervals overlap however
    well the model routes. A campaign performing at its theoretical optimum still
    reports False. Do NOT report `converged: False` to the user as "not learning".
    Judge a contextual campaign by its reward rate over time and by traffic
    splitting across arms; traffic collapsing onto a single arm is the sign the
    model has stopped using the context.

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
    except ValueError as e:
        return f"Invalid input: {str(e)}"
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
    except ValueError as e:
        return f"Invalid input: {str(e)}"
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
    except ValueError as e:
        return f"Invalid input: {str(e)}"
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
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except BanditDBError as e:
        return f"Error restoring campaign: {str(e)}"


@mcp.tool()
def campaign_budget(campaign_id: str) -> str:
    """
    Show how a budget-constrained campaign is spending its budgets.

    Use this to answer: "How much budget is left? Is spend on pace? Why does the
    campaign stop suggesting the expensive option?"

    Per resource it reports budget, consumed, remaining, utilization, and the
    shadow price (lambda): what one unit of the resource currently costs an
    action's score. Rising lambda means spend is running ahead of pace; 0 means the
    budget is not limiting anything. Once a resource is exhausted, every action that
    costs it stays unavailable for good — budgets do not reset.

    Args:
        campaign_id: The campaign to inspect.

    Returns:
        A per-resource budget summary, or a note that the campaign has no budgets.
    """
    try:
        p = db.pacing(campaign_id)
        if p is None:
            return f"Campaign '{campaign_id}' has no budget constraints."
        lines = [
            f"Campaign: {campaign_id} "
            f"({'adaptive' if p.get('adaptive') else 'uniform'} pacing)",
            "",
            "Budgets:",
        ]
        for r in p.get("resources", []):
            lines.append(
                f"  • {r['name']}: {r['consumed']:g} / {r['budget']:g} spent "
                f"({r['utilization']:.0%}), remaining={r['remaining']:g}, "
                f"lambda={r['lambda']:.4f}, decisions={r['decisions']}/{r['horizon']}"
                + ("  ⚠ EXHAUSTED" if r.get("is_exhausted") else "")
            )
        return "\n".join(lines)
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except BanditDBError as e:
        return f"Error fetching campaign budget: {str(e)}"


@mcp.tool()
def add_arm(
    campaign_id: str,
    arm_id: str,
    group: str | None = None,
    warm_start_from: str | None = None,
    costs: dict[str, float] | None = None,
) -> str:
    """
    Add a new action (arm) to an existing campaign without losing what it has learned.

    Args:
        campaign_id: The campaign to extend.
        arm_id: Name of the new action. Must not already exist.
        group: Optional category label (e.g. 'shoes'). Arms in the same group can
               lend their knowledge to a new member.
        warm_start_from: Optional. "population" starts the new arm from the average
               of all active arms; "group" from the average of its group. Default is
               a cold start that learns from scratch.
        costs: Budget campaigns only. What the new action consumes per suggestion,
               keyed by budget name, e.g. {"promo_budget": 3.0}. Omitted = free.

    Returns:
        Confirmation or error message.
    """
    try:
        warm_start = {"from": warm_start_from} if warm_start_from else None
        db.add_arm(campaign_id, arm_id, group=group, warm_start=warm_start, costs=costs)
        return f"Arm '{arm_id}' added to campaign '{campaign_id}'."
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except BanditDBError as e:
        return f"Error adding arm: {str(e)}"


@mcp.tool()
def set_arm_status(campaign_id: str, arm_id: str, status: str) -> str:
    """
    Pause, retire, or reactivate an action (arm) in a campaign.

    Use "paused" for temporary unavailability (out of stock, seasonal), "retired" for
    actions that should not come back, and "active" to restore either. The arm keeps
    everything it learned in every state. To skip an arm for one request only, pass
    exclude_arms to get_intuition instead.

    Args:
        campaign_id: The campaign the arm belongs to.
        arm_id: The arm to change.
        status: "active", "paused", or "retired".

    Returns:
        Confirmation or error message.
    """
    try:
        db.set_arm_status(campaign_id, arm_id, status)
        return f"Arm '{arm_id}' in campaign '{campaign_id}' is now {status}."
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except BanditDBError as e:
        return f"Error setting arm status: {str(e)}"


@mcp.tool()
def get_intuition(
    campaign_id: str,
    context: list[float],
    exclude_arms: list[str] | None = None,
    eligible_arms: list[str] | None = None,
) -> str:
    """
    Ask the BanditDB Hive Mind for the best strategy or action to take.

    Args:
        campaign_id: The ID of the decision campaign (e.g., 'llm_routing', 'support_strategy').
        context: A list of finite floats representing the current state (e.g., user
            sentiment, task difficulty). Keep values on a similar, modest scale —
            roughly -1 to 1 works well. Very large values are rejected.
        exclude_arms: Optional. Actions not allowed for this request only (e.g.
            already tried, unavailable right now). Does not change the campaign.
        eligible_arms: Optional. Choose only among these actions for this request.

    Returns:
        A string telling you which action to take, and the interaction_id you MUST save for the reward.
    """
    try:
        arm_id, interaction_id = db.predict(
            campaign_id, context,
            eligible_arms=eligible_arms, exclude_arms=exclude_arms,
        )
        return (
            f"💡 BanditDB Suggests: Take action '{arm_id}'.\n"
            f"[IMPORTANT] Save this interaction_id for the outcome: {interaction_id}"
        )
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except BanditDBError as e:
        return f"Error connecting to Hive Mind: {str(e)}"

@mcp.tool()
def record_outcome(interaction_id: str, reward: float) -> str:
    """
    Tell the BanditDB Hive Mind if the strategy it suggested was successful.

    Args:
        interaction_id: The unique ID returned by get_intuition.
        reward: A number between 0.0 and 1.0 — 1.0 for a clear success, 0.0 for a
            failure, and values in between for partial success. Anything outside
            this range is rejected, so scale it rather than passing a raw score.

    Returns:
        Confirmation that the global math matrices have been updated.
    """
    try:
        success = db.reward(interaction_id, reward)
        if success:
            return "🧠 Mathematical weights updated! The Swarm has learned from this interaction."
        return "Failed to update weights."
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except BanditDBError as e:
        return f"Error recording outcome: {str(e)}"

def main():
    """Entry point for the command-line interface."""
    #print("🚀 Starting BanditDB MCP Server...", flush=True)
    mcp.run()

if __name__ == "__main__":
    main()
