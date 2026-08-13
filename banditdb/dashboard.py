"""
banditdb-dashboard — terminal dashboard for BanditDB administrators.

Usage:
    banditdb-dashboard
    banditdb-dashboard --url http://localhost:8080 --refresh 5
    BANDITDB_URL=http://host:8080 BANDITDB_API_KEY=key banditdb-dashboard
"""
from __future__ import annotations

import argparse
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, Rule, Static

from banditdb import BanditDBError, Client


# ── helpers ──────────────────────────────────────────────────────────────────

def _color(text: str, color: str) -> str:
    return f"[{color}]{text}[/{color}]"

def _entropy_style(status: str) -> Tuple[str, str]:
    """Returns (rich_color, symbol) for an entropy status string."""
    if status == "ok":
        return "#22C55E", "✓"
    if status == "warning":
        return "#EAB308", "⚠"
    return "#EF4444", "✗"

def _conv_style(converged) -> Tuple[str, str]:
    if converged is True:
        return "#22C55E", "converged"
    if converged is False:
        return "#60A5FA", "running"
    return "#606060", "no data"

def _algo_name(algo) -> str:
    """Normalise algorithm field — server may return a string or a dict enum."""
    if isinstance(algo, dict):
        key = next(iter(algo), "linucb")
        return key.lower()
    return str(algo).lower()

def _algo_color(algo) -> str:
    colors = {
        "linucb":         "#60A5FA",
        "thompson":       "#A78BFA",
        "neural_lin_ucb": "#F472B6",
        "neural":         "#F472B6",
        "progressive":    "#FB923C",
        "tournament":     "#FB923C",
    }
    return colors.get(_algo_name(algo), "#A0A0A0")

def _parse_prometheus(text: str) -> Dict[str, str]:
    """Parse flat Prometheus text into {metric_name: value} for scalar gauges/counters."""
    out: Dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out

def _fmt_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

def _reward_gap_pct(preds: int, rewards: int) -> str:
    if preds == 0:
        return "—"
    gap = (preds - rewards) / preds * 100
    if gap > 10:
        return f"[#EF4444]{gap:.1f}%[/#EF4444]"
    if gap > 3:
        return f"[#EAB308]{gap:.1f}%[/#EAB308]"
    return f"[#22C55E]{gap:.1f}%[/#22C55E]"


# ── simulation helpers ─────────────────────────────────────────────────────────

_SPARKS = "▁▂▃▄▅▆▇█"
_SIM_ARM_COLORS = ["#60A5FA", "#A78BFA", "#34D399", "#F472B6", "#FB923C", "#FACC15"]


def _sparkline(values: List[float], width: int = 32, vmin: float = 0.0, vmax: float = 1.0) -> str:
    if not values:
        return "─" * width
    n = len(values)
    if n <= width:
        sampled = list(values) + [values[-1]] * (width - n)
    else:
        sampled = [values[int(i * (n - 1) / (width - 1))] for i in range(width)]
    span = vmax - vmin
    if span == 0:
        return "▄" * width
    def _b(v: float) -> str:
        return _SPARKS[max(0, min(7, int(max(0.0, min(1.0, (v - vmin) / span)) * 7)))]
    return "".join(_b(v) for v in sampled)


def _ci_bar(lo: Optional[float], hi: Optional[float], mean: Optional[float], width: int = 26) -> str:
    if lo is None or hi is None or mean is None:
        return "[" + "─" * width + "]"
    lo   = max(0.0, min(1.0, lo))
    hi   = max(0.0, min(1.0, hi))
    mean = max(0.0, min(1.0, mean))
    bar  = ["─"] * width
    lo_p, hi_p, mid_p = int(lo * (width - 1)), int(hi * (width - 1)), int(mean * (width - 1))
    for i in range(lo_p, hi_p + 1):
        if 0 <= i < width:
            bar[i] = "─"
    if 0 <= lo_p  < width: bar[lo_p]  = "├"
    if 0 <= hi_p  < width: bar[hi_p]  = "┤"
    if 0 <= mid_p < width: bar[mid_p] = "●"
    return "[" + "".join(bar) + "]"


def _sim_arm_color(idx: int) -> str:
    return _SIM_ARM_COLORS[idx % len(_SIM_ARM_COLORS)]


# ── data fetch ────────────────────────────────────────────────────────────────

def fetch_overview(db: Client) -> dict:
    """Fetch all data needed for the main screen. Runs in a thread."""
    result: dict = {
        "error":      None,
        "campaigns":  [],
        "metrics":    {},
        "alerts":     [],
        "fetch_time": time.time(),
    }
    try:
        raw_campaigns = db.list_campaigns()
        campaigns_list = raw_campaigns if isinstance(raw_campaigns, list) else []

        rows = []
        for c in campaigns_list:
            cid = c.get("campaign_id", "")
            if c.get("archived"):
                continue
            try:
                report = db.report(cid)
            except Exception:
                report = {}
            try:
                diag = db.diagnostics(cid)
            except Exception:
                diag = {}

            total_preds   = report.get("total_predictions", 0) or 0
            total_rewards = report.get("total_rewards", 0) or 0
            entropy_status = diag.get("entropy_status", "unknown")
            entropy_trend  = diag.get("entropy_trend", "unknown")
            converged      = report.get("converged")
            leading_arm    = report.get("leading_arm") or "—"
            algo           = c.get("algorithm", "linucb")
            arm_count      = c.get("arm_count", 0)
            likely_cause   = diag.get("likely_cause")
            suggested_action = diag.get("suggested_action")

            if entropy_status in ("warning", "critical") and likely_cause:
                result["alerts"].append({
                    "campaign": cid,
                    "status":   entropy_status,
                    "cause":    likely_cause,
                    "action":   suggested_action or "",
                })

            rows.append({
                "id":             cid,
                "algorithm":      algo,
                "arm_count":      arm_count,
                "alpha":          c.get("alpha", 1.0),
                "total_preds":    total_preds,
                "total_rewards":  total_rewards,
                "entropy_status": entropy_status,
                "entropy_trend":  entropy_trend,
                "converged":      converged,
                "leading_arm":    leading_arm,
                "report":         report,
                "diag":           diag,
            })

        result["campaigns"] = rows

        # Prometheus metrics — /metrics returns text/plain, use session directly
        try:
            r = db.session.get(f"{db.url}/metrics", timeout=5)
            result["metrics"] = _parse_prometheus(r.text)
        except Exception:
            result["metrics"] = {}

    except BanditDBError as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = f"Connection error: {e}"
    return result


def fetch_campaign_detail(db: Client, campaign_id: str) -> dict:
    """Fetch deep data for one campaign. Runs in a thread."""
    out: dict = {"id": campaign_id, "error": None}
    try:
        out["report"] = db.report(campaign_id)
    except Exception as e:
        out["report"] = {}
        out["error"] = str(e)
    try:
        out["diag"] = db.diagnostics(campaign_id)
    except Exception:
        out["diag"] = {}
    try:
        raw = db._get(f"/campaign/{campaign_id}")
        out["info"] = raw
    except Exception:
        out["info"] = {}
    return out


# ── CSS ───────────────────────────────────────────────────────────────────────

DASHBOARD_CSS = """
Screen {
    background: #0A0A0A;
}

#header-bar {
    height: 3;
    background: #111111;
    border-bottom: solid #2A2A2A;
    padding: 0 2;
    layout: horizontal;
    align: left middle;
}

.header-stat {
    color: #A0A0A0;
    margin-right: 3;
}

.header-stat-val {
    color: #F5F5F5;
    text-style: bold;
}

#stat-tiles {
    height: 5;
    layout: horizontal;
    padding: 1 2 0 2;
}

.stat-tile {
    background: #111111;
    border: solid #2A2A2A;
    padding: 0 2;
    margin-right: 1;
    min-width: 18;
    height: 3;
    align: left middle;
}

.stat-label {
    color: #606060;
    text-style: bold;
}

.stat-value {
    color: #F5F5F5;
    text-style: bold;
}

#section-campaigns {
    margin: 1 2 0 2;
    color: #606060;
    text-style: bold;
}

#campaign-table {
    margin: 0 2;
    height: 1fr;
    border: solid #2A2A2A;
    background: #111111;
}

#alerts-container {
    height: auto;
    max-height: 6;
    margin: 0 2 1 2;
    border: solid #2A2A2A;
    background: #111111;
    padding: 0 1;
}

.alert-label {
    color: #EAB308;
}

.alert-critical {
    color: #EF4444;
}

#status-bar {
    height: 1;
    background: #111111;
    border-top: solid #2A2A2A;
    padding: 0 2;
    color: #606060;
}

/* Detail screen */
#detail-header {
    height: 3;
    background: #111111;
    border-bottom: solid #2A2A2A;
    padding: 0 2;
    align: left middle;
    color: #F5F5F5;
}

#detail-body {
    layout: horizontal;
    height: 1fr;
}

#detail-left {
    width: 1fr;
    border-right: solid #2A2A2A;
    padding: 1 2;
}

#detail-right {
    width: 1fr;
    padding: 1 2;
}

.detail-section-title {
    color: #F97316;
    text-style: bold;
    margin-bottom: 1;
}

.detail-arm-table {
    height: auto;
    max-height: 14;
    border: solid #2A2A2A;
    background: #0A0A0A;
    margin-bottom: 1;
}

.detail-kv {
    color: #A0A0A0;
    margin-bottom: 0;
}

Footer {
    background: #111111;
    color: #606060;
    border-top: solid #2A2A2A;
}

/* Simulation screen */
#sim-header {
    height: 2;
    background: #111111;
    border-bottom: solid #2A2A2A;
    padding: 0 2;
    align: left middle;
    color: #F5F5F5;
}

#sim-body {
    layout: horizontal;
    height: 1fr;
}

#sim-left {
    width: 1fr;
    border-right: solid #2A2A2A;
    padding: 1 2;
}

#sim-right {
    width: 1fr;
    padding: 1 2;
}

.sim-title {
    color: #A78BFA;
    text-style: bold;
    margin-top: 1;
    margin-bottom: 1;
}

#sim-bar {
    height: 1;
    background: #111111;
    border-top: solid #2A2A2A;
    padding: 0 2;
    color: #606060;
}
"""


# ── detail screen ─────────────────────────────────────────────────────────────

class CampaignDetailScreen(Screen):
    BINDINGS = [
        Binding("escape,q", "app.pop_screen", "Back"),
        Binding("r", "refresh_detail", "Refresh"),
    ]

    def __init__(self, db: Client, campaign_id: str, **kwargs):
        super().__init__(**kwargs)
        self._db = db
        self._campaign_id = campaign_id
        self._data: dict = {}
        self._executor = ThreadPoolExecutor(max_workers=1)

    def compose(self) -> ComposeResult:
        yield Static(f"← Back  [bold #F5F5F5]{self._campaign_id}[/bold #F5F5F5]", id="detail-header")
        with Horizontal(id="detail-body"):
            with ScrollableContainer(id="detail-left"):
                yield Static("ARM PERFORMANCE", classes="detail-section-title")
                yield DataTable(id="arm-table", classes="detail-arm-table")
                yield Static("REWARD PIPELINE", classes="detail-section-title")
                yield Static("Loading…", id="pipeline-stats")
            with ScrollableContainer(id="detail-right"):
                yield Static("LEARNING HEALTH", classes="detail-section-title")
                yield Static("Loading…", id="health-stats")
                yield Static("CONVERGENCE", classes="detail-section-title")
                yield Static("Loading…", id="conv-stats")
                yield Static("THETA NORMS", classes="detail-section-title")
                yield Static("Loading…", id="theta-stats")
        yield Footer()

    def on_mount(self) -> None:
        tbl: DataTable = self.query_one("#arm-table", DataTable)
        tbl.add_columns("Arm", "Preds", "Rewards", "Mean Rwd", "CI Low", "CI High", "Traffic %")
        tbl.cursor_type = "row"
        self.action_refresh_detail()

    def action_refresh_detail(self) -> None:
        future = self._executor.submit(fetch_campaign_detail, self._db, self._campaign_id)
        self.set_timer(0.05, lambda: self._poll_future(future))

    def _poll_future(self, future) -> None:
        if future.done():
            self._data = future.result()
            self._render_detail()
        else:
            self.set_timer(0.1, lambda: self._poll_future(future))

    def _render_detail(self) -> None:
        report = self._data.get("report", {})
        diag   = self._data.get("diag", {})
        info   = self._data.get("info", {})

        algo      = info.get("algorithm", "linucb")
        alpha     = info.get("alpha", 1.0)
        arm_count = len(info.get("arms", {}))

        # Update header
        algo_color = _algo_color(algo)
        algo_str   = _algo_name(algo)
        hdr = self.query_one("#detail-header", Static)
        hdr.update(
            f"← [dim]Back (Esc)[/dim]  [bold #F5F5F5]{self._campaign_id}[/bold #F5F5F5]  "
            f"[{algo_color}]{algo_str}[/{algo_color}] · α={alpha} · {arm_count} arms"
        )

        # Arm table
        tbl: DataTable = self.query_one("#arm-table", DataTable)
        tbl.clear()
        arm_stats = report.get("arms", {})
        for arm_id, s in sorted(arm_stats.items()):
            preds    = s.get("predictions", 0) or 0
            rewards  = s.get("rewards", 0) or 0
            mean     = s.get("mean_reward")
            lo       = s.get("lower_ci")
            hi       = s.get("upper_ci")
            traffic  = s.get("traffic_share", 0) or 0

            mean_s  = f"{mean:.3f}" if mean is not None else "—"
            lo_s    = f"{lo:.3f}"   if lo   is not None else "—"
            hi_s    = f"{hi:.3f}"   if hi   is not None else "—"
            traffic_s = f"{traffic*100:.1f}%"

            # Color mean reward
            if mean is not None:
                if mean >= 0.7:
                    mean_s = f"[#22C55E]{mean_s}[/#22C55E]"
                elif mean >= 0.4:
                    mean_s = f"[#EAB308]{mean_s}[/#EAB308]"
                else:
                    mean_s = f"[#EF4444]{mean_s}[/#EF4444]"

            tbl.add_row(arm_id, str(preds), str(rewards), mean_s, lo_s, hi_s, traffic_s)

        # Pipeline stats
        total_preds   = report.get("total_predictions", 0) or 0
        total_rewards = report.get("total_rewards", 0) or 0
        gap_pct       = _reward_gap_pct(total_preds, total_rewards)
        pipeline = self.query_one("#pipeline-stats", Static)
        pipeline.update(
            f"[#A0A0A0]Predictions:[/#A0A0A0] [#F5F5F5]{_fmt_num(total_preds)}[/#F5F5F5]\n"
            f"[#A0A0A0]Rewards:    [/#A0A0A0] [#F5F5F5]{_fmt_num(total_rewards)}[/#F5F5F5]\n"
            f"[#A0A0A0]Gap:        [/#A0A0A0] {gap_pct}"
        )

        # Health stats
        entropy_status = diag.get("entropy_status", "unknown")
        entropy_trend  = diag.get("entropy_trend", "unknown")
        entropy_val    = diag.get("selection_entropy")
        likely_cause   = diag.get("likely_cause", "")
        suggested      = diag.get("suggested_action", "")
        ec, es         = _entropy_style(entropy_status)
        trend_color = {
            "falling":   "#EF4444",
            "recovering":"#22C55E",
            "stable":    "#A0A0A0",
        }.get(entropy_trend, "#606060")
        neural_buf = diag.get("neural_buffer_size")
        neural_line = (
            f"\n[#A0A0A0]Neural buf: [/#A0A0A0][#F5F5F5]{neural_buf}[/#F5F5F5]"
            if neural_buf is not None else ""
        )
        chall_pct = diag.get("challenger_traffic_pct")
        tourney_wins = diag.get("tournament_win_streak")
        tourney_line = ""
        if chall_pct is not None:
            tourney_line = (
                f"\n[#A0A0A0]Challenger: [/#A0A0A0][#FB923C]{chall_pct:.1f}%[/#FB923C]"
                f"  [#A0A0A0]Wins:[/#A0A0A0] [#F5F5F5]{tourney_wins}[/#F5F5F5]"
            )
        entropy_s = f"{entropy_val:.3f}" if entropy_val is not None else "—"
        health_text = (
            f"[#A0A0A0]Entropy:[/#A0A0A0] [#F5F5F5]{entropy_s}[/#F5F5F5]  "
            f"[{ec}]{es} {entropy_status}[/{ec}]\n"
            f"[#A0A0A0]Trend:  [/#A0A0A0] [{trend_color}]{entropy_trend}[/{trend_color}]"
        )
        if likely_cause:
            health_text += (
                f"\n[#A0A0A0]Cause:  [/#A0A0A0] [#EAB308]{likely_cause}[/#EAB308]"
                f"\n[#A0A0A0]Action: [/#A0A0A0] [#F97316]{suggested}[/#F97316]"
            )
        health_text += neural_line + tourney_line
        self.query_one("#health-stats", Static).update(health_text)

        # Convergence stats
        converged   = report.get("converged")
        leading_arm = report.get("leading_arm") or "—"
        cc, cs      = _conv_style(converged)
        arm_stats_l = list(report.get("arms", {}).items())
        best_mean_s = "—"
        if arm_stats_l and leading_arm != "—":
            best = report.get("arms", {}).get(leading_arm, {})
            m = best.get("mean_reward")
            if m is not None:
                best_mean_s = f"{m:.3f}"
        conv_text = (
            f"[#A0A0A0]Status: [/#A0A0A0] [{cc}]{cs}[/{cc}]\n"
            f"[#A0A0A0]Leader: [/#A0A0A0] [#F5F5F5]{leading_arm}[/#F5F5F5]  "
            f"[#A0A0A0]Mean:[/#A0A0A0] [#22C55E]{best_mean_s}[/#22C55E]"
        )
        self.query_one("#conv-stats", Static).update(conv_text)

        # Theta norms
        arm_diags = diag.get("arm_stats", {}) or diag.get("arms", {}) or {}
        if arm_diags:
            lines = []
            for arm_id, ad in sorted(arm_diags.items()):
                theta = ad.get("theta_norm", 0)
                a_max = ad.get("a_inv_diag_max", 0)
                theta_color = "#22C55E" if theta > 0.5 else ("#EAB308" if theta > 0 else "#EF4444")
                lines.append(
                    f"[#A0A0A0]{arm_id[:16]:<16}[/#A0A0A0]  "
                    f"θ=[{theta_color}]{theta:.3f}[/{theta_color}]  "
                    f"[#606060]A⁻¹=[/#606060][#60A5FA]{a_max:.4f}[/#60A5FA]"
                )
            self.query_one("#theta-stats", Static).update("\n".join(lines))
        else:
            self.query_one("#theta-stats", Static).update("[#606060]No arm diagnostics available[/#606060]")


# ── simulation screen ─────────────────────────────────────────────────────────

class SimulationScreen(Screen):
    """
    Runs a synthetic predict/reward loop on a cloned campaign and visualises
    the full cold-start → convergence learning arc in real time.

    Creates _sim_<campaign_id>, runs batches of random-context predictions with
    configurable true reward rates, plots entropy decay, arm concentration, theta
    norm growth, and CI narrowing. Auto-deletes the sim campaign on exit.
    """

    BINDINGS = [
        Binding("escape,q", "back", "Back"),
        Binding("space",    "toggle_sim",  "Start/Stop"),
        Binding("r",        "reset_sim",   "Reset"),
        Binding("plus",     "speed_up",    "+Speed"),
        Binding("minus",    "slow_down",   "-Speed"),
    ]

    _SIM_PREFIX = "_sim_"

    def __init__(self, db: Client, campaign_id: str, **kwargs):
        super().__init__(**kwargs)
        self._source_id = campaign_id
        self._sim_id    = f"{self._SIM_PREFIX}{campaign_id}"
        self._info_db   = db  # original client for setup (reliable timeout)
        # Dedicated client with short timeout so failed predict calls don't stall batches
        self._db = Client(
            url=db.url,
            timeout=0.5,
            max_retries=0,
            api_key=db.session.headers.get("X-Api-Key"),
        )

        # Populated after info fetch
        self._feature_dim:  int              = 3
        self._arms:         List[str]        = []
        self._arm_colors:   Dict[str, str]   = {}
        self._true_rewards: Dict[str, float] = {}
        self._raw_algo:     Any              = "linucb"
        self._alpha:        float            = 1.0

        # Sim state
        self._sim_created  = False
        self._running      = False
        self._tick_busy    = False
        self._step         = 0
        self._batch        = 10
        self._sim_timer    = None

        # History (one entry per tick)
        self._entropy_hist: List[float]           = []
        self._traffic_hist: Dict[str, List[float]] = {}
        self._theta_hist:   Dict[str, List[float]] = {}
        self._ci_snap:      Dict[str, dict]        = {}

        self._executor = ThreadPoolExecutor(max_workers=1)

    # ── layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Static("", id="sim-header")
        with Horizontal(id="sim-body"):
            with ScrollableContainer(id="sim-left"):
                yield Static("ENTROPY  [dim]uncertainty → confidence[/dim]",            classes="sim-title")
                yield Static("[#606060]waiting…[/#606060]",                              id="entropy-chart")
                yield Rule()
                yield Static("ARM CONCENTRATION  [dim]cold-start → dominant arm[/dim]", classes="sim-title")
                yield Static("",                                                          id="concentration-chart")
                yield Rule()
                yield Static("SIMULATION SETUP",                                          classes="sim-title")
                yield Static("",                                                          id="sim-config")
            with ScrollableContainer(id="sim-right"):
                yield Static("THETA NORMS  [dim]weight growth = learning[/dim]",          classes="sim-title")
                yield Static("",                                                           id="theta-chart")
                yield Rule()
                yield Static("CONFIDENCE INTERVALS  [dim]narrowing = confident[/dim]",    classes="sim-title")
                yield Static("",                                                           id="ci-chart")
                yield Rule()
                yield Static("LEARNING PHASE",                                             classes="sim-title")
                yield Static("",                                                           id="phase-panel")
        yield Static("", id="sim-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._update_header()
        self._update_bar("Loading campaign…")
        f = self._executor.submit(self._fetch_info)
        self.set_timer(0.1, lambda: self._poll_info(f))

    # ── setup ─────────────────────────────────────────────────────────────────

    def _fetch_info(self) -> dict:
        try:
            return {"ok": True, "info": self._info_db._get(f"/campaign/{self._source_id}")}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _poll_info(self, f) -> None:
        if not f.done():
            self.set_timer(0.1, lambda: self._poll_info(f))
            return
        r = f.result()
        if not r["ok"]:
            self._update_bar(f"[#EF4444]Error: {r['error']}[/#EF4444]")
            return
        info = r["info"]
        self._feature_dim = info.get("feature_dim", 3)
        self._raw_algo    = info.get("algorithm", "linucb")
        self._alpha       = info.get("alpha", 1.0)
        arms_raw = info.get("arms", {})
        self._arms = sorted(arms_raw.keys()) if isinstance(arms_raw, dict) else [str(a) for a in arms_raw]
        n = len(self._arms)
        for i, arm in enumerate(self._arms):
            self._arm_colors[arm]    = _sim_arm_color(i)
            self._true_rewards[arm]  = round(0.70 - i * (0.40 / max(n - 1, 1)), 2)
        self._render_config()
        self._update_header()
        self._update_bar(
            "[#A0A0A0][Space] start · [+/-] speed · [R] reset · [Esc] back[/#A0A0A0]"
        )

    def _render_config(self) -> None:
        ac    = _algo_color(self._raw_algo)
        aname = _algo_name(self._raw_algo)
        lines = [
            f"[#A0A0A0]Algorithm  [/#A0A0A0][{ac}]{aname}[/{ac}]  "
            f"[#A0A0A0]α=[/#A0A0A0][#F5F5F5]{self._alpha}[/#F5F5F5]  "
            f"[#A0A0A0]dim=[/#A0A0A0][#F5F5F5]{self._feature_dim}[/#F5F5F5]",
            "",
            "[#606060]Simulated true reward rates (arm 0 = best):[/#606060]",
        ]
        for arm in self._arms:
            rate  = self._true_rewards[arm]
            color = self._arm_colors[arm]
            f_    = int(rate * 12)
            bar   = f"[{color}]{'█' * f_}[/{color}][#2A2A2A]{'░' * (12 - f_)}[/#2A2A2A]"
            lines.append(f"  [{color}]{arm:<14}[/{color}]  {bar}  [#F5F5F5]{rate:.2f}[/#F5F5F5]")
        self.query_one("#sim-config", Static).update("\n".join(lines))

    # ── actions ───────────────────────────────────────────────────────────────

    def action_back(self) -> None:
        self._running = False
        if self._sim_timer:
            self._sim_timer.stop()
        if self._sim_created:
            self._executor.submit(lambda: self._info_db.delete_campaign(self._sim_id))
        self.app.pop_screen()

    def action_toggle_sim(self) -> None:
        if not self._arms:
            return
        if self._running:
            self._running = False
            if self._sim_timer:
                self._sim_timer.stop()
                self._sim_timer = None
            self._update_bar("[#EAB308]Paused[/#EAB308] — Space to resume")
        elif self._sim_created:
            self._start_ticking()
        else:
            self._update_bar("[#A0A0A0]Creating sim campaign…[/#A0A0A0]")
            f = self._executor.submit(self._create_campaign)
            self.set_timer(0.1, lambda: self._poll_create(f))

    def action_reset_sim(self) -> None:
        self._running = False
        if self._sim_timer:
            self._sim_timer.stop()
            self._sim_timer = None
        if self._sim_created:
            self._executor.submit(lambda: self._info_db.delete_campaign(self._sim_id))
            self._sim_created = False
        self._step = 0
        self._tick_busy = False
        self._entropy_hist.clear()
        self._traffic_hist.clear()
        self._theta_hist.clear()
        self._ci_snap.clear()
        for wid in ("#entropy-chart", "#concentration-chart", "#theta-chart", "#ci-chart", "#phase-panel"):
            self.query_one(wid, Static).update("[#606060]—[/#606060]")
        self._update_header()
        self._update_bar("[#A0A0A0]Reset — Space to start new run[/#A0A0A0]")

    def action_speed_up(self) -> None:
        self._batch = min(200, int(self._batch * 1.5) + 1)
        self._update_header()

    def action_slow_down(self) -> None:
        self._batch = max(5, int(self._batch / 1.5))
        self._update_header()

    # ── sim lifecycle ─────────────────────────────────────────────────────────

    def _create_campaign(self) -> dict:
        try:
            try: self._info_db.delete_campaign(self._sim_id)
            except Exception: pass
            self._info_db._post("/campaign", {
                "campaign_id": self._sim_id,
                "arms":        self._arms,
                "feature_dim": self._feature_dim,
                "alpha":       self._alpha,
                "algorithm":   self._raw_algo,
            })
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _poll_create(self, f) -> None:
        if not f.done():
            self.set_timer(0.1, lambda: self._poll_create(f))
            return
        r = f.result()
        if not r["ok"]:
            self._update_bar(f"[#EF4444]Create failed: {r['error']}[/#EF4444]")
            return
        self._sim_created = True
        self._start_ticking()

    def _start_ticking(self) -> None:
        self._running   = True
        self._sim_timer = self.set_interval(0.25, self._on_tick)
        self._update_header()

    def _on_tick(self) -> None:
        if not self._running or self._tick_busy:
            return
        self._tick_busy = True
        f = self._executor.submit(self._run_batch, self._batch)
        self.set_timer(0.05, lambda: self._poll_batch(f))

    def _run_batch(self, n: int) -> dict:
        for _ in range(n):
            ctx = [random.gauss(0, 1) for _ in range(self._feature_dim)]
            try:
                arm_id, iid = self._db.predict(self._sim_id, ctx)
                p = self._true_rewards.get(arm_id, 0.5)
                r = 1.0 if random.random() < p else 0.0
                self._db.reward(iid, r)
            except Exception:
                pass
        try:    report = self._db.report(self._sim_id)
        except: report = {}
        try:    diag   = self._db.diagnostics(self._sim_id)
        except: diag   = {}
        return {"report": report, "diag": diag}

    def _poll_batch(self, f) -> None:
        if not f.done():
            self.set_timer(0.05, lambda: self._poll_batch(f))
            return
        self._tick_busy = False
        if not self._running:
            return
        self._step += self._batch
        r = f.result()
        self._ingest(r["report"], r["diag"])
        self._render_all(r["report"], r["diag"])
        self._update_header()

    def _ingest(self, report: dict, diag: dict) -> None:
        entropy = diag.get("selection_entropy")
        if entropy is not None:
            self._entropy_hist.append(entropy)
        for arm_id, s in report.get("arms", {}).items():
            self._traffic_hist.setdefault(arm_id, []).append(s.get("traffic_share", 0) or 0)
        arm_diags = diag.get("arm_stats", {}) or diag.get("arms", {}) or {}
        for arm_id, ad in arm_diags.items():
            self._theta_hist.setdefault(arm_id, []).append(ad.get("theta_norm", 0) or 0)
        for arm_id, s in report.get("arms", {}).items():
            self._ci_snap[arm_id] = {
                "lo":    s.get("lower_ci"),
                "hi":    s.get("upper_ci"),
                "mean":  s.get("mean_reward"),
                "preds": s.get("predictions", 0) or 0,
            }

    # ── rendering ─────────────────────────────────────────────────────────────

    def _update_header(self) -> None:
        if self._running:
            state = "[#22C55E]▶ RUNNING[/#22C55E]"
        elif self._step > 0:
            state = "[#EAB308]⏸ PAUSED[/#EAB308]"
        else:
            state = "[#606060]READY[/#606060]"
        self.query_one("#sim-header", Static).update(
            f"[#F97316]SIM[/#F97316]  [bold #F5F5F5]{self._source_id}[/bold #F5F5F5]  "
            f"{state}  [#F5F5F5]{self._step:,}[/#F5F5F5] [#606060]steps · {self._batch}/tick[/#606060]"
        )

    def _update_bar(self, msg: str) -> None:
        try:
            self.query_one("#sim-bar", Static).update(msg)
        except Exception:
            pass

    def _render_all(self, report: dict, diag: dict) -> None:
        self._render_entropy()
        self._render_concentration()
        self._render_theta()
        self._render_ci()
        self._render_phase(report, diag)

    def _render_entropy(self) -> None:
        h = self._entropy_hist
        if not h:
            return
        spark = _sparkline(h, width=40)
        cur   = h[-1]
        ec    = "#22C55E" if cur > 0.6 else ("#EAB308" if cur > 0.3 else "#60A5FA")
        label = "exploring" if cur > 0.6 else ("focusing" if cur > 0.3 else "converged")
        self.query_one("#entropy-chart", Static).update(
            f"[#606060]1.0 ╷[/#606060]\n"
            f"[#606060]    │[/#606060][#A78BFA]{spark}[/#A78BFA]\n"
            f"[#606060]0.0 ╵[/#606060]  [{ec}]{cur:.3f}  {label}[/{ec}]  "
            f"[#606060]n={len(h)}[/#606060]"
        )

    def _render_concentration(self) -> None:
        if not self._traffic_hist:
            return
        lines = []
        for arm in self._arms:
            hist  = self._traffic_hist.get(arm, [])
            cur   = hist[-1] if hist else 0.0
            color = self._arm_colors.get(arm, "#A0A0A0")
            spark = _sparkline(hist, width=18) if hist else "─" * 18
            f_    = int(cur * 14)
            bar   = f"[{color}]{'█' * f_}[/{color}][#2A2A2A]{'░' * (14 - f_)}[/#2A2A2A]"
            lines.append(
                f"[{color}]{arm:<12}[/{color}] [#404040]{spark}[/#404040] {bar} "
                f"[#F5F5F5]{cur * 100:5.1f}%[/#F5F5F5]"
            )
        self.query_one("#concentration-chart", Static).update("\n".join(lines))

    def _render_theta(self) -> None:
        if not self._theta_hist:
            self.query_one("#theta-chart", Static).update(
                "[#606060]No theta data yet — needs more steps[/#606060]"
            )
            return
        all_vals = [v for h in self._theta_hist.values() for v in h]
        vmax     = max(all_vals) if all_vals else 1.0
        lines    = []
        for arm in self._arms:
            hist  = self._theta_hist.get(arm, [])
            if not hist:
                continue
            spark = _sparkline(hist, width=28, vmin=0.0, vmax=max(vmax, 0.001))
            cur   = hist[-1]
            color = self._arm_colors.get(arm, "#A0A0A0")
            tc    = "#22C55E" if cur > 0.5 else ("#EAB308" if cur > 0.1 else "#606060")
            lines.append(
                f"[{color}]{arm:<12}[/{color}] [#383838]{spark}[/#383838] [{tc}]{cur:.3f}[/{tc}]"
            )
        self.query_one("#theta-chart", Static).update(
            "\n".join(lines) if lines else "[#606060]—[/#606060]"
        )

    def _render_ci(self) -> None:
        if not self._ci_snap:
            return
        lines = []
        for arm in self._arms:
            s     = self._ci_snap.get(arm, {})
            lo, hi, mean = s.get("lo"), s.get("hi"), s.get("mean")
            preds = s.get("preds", 0)
            color = self._arm_colors.get(arm, "#A0A0A0")
            w     = (hi - lo) if (lo is not None and hi is not None) else None
            cc    = "#22C55E" if (w is not None and w < 0.10) else (
                    "#EAB308" if (w is not None and w < 0.30) else "#EF4444")
            label = ("confident" if (w is not None and w < 0.10) else
                     "learning"  if (w is not None and w < 0.30) else "uncertain")
            bar   = _ci_bar(lo, hi, mean)
            w_s   = f"±{w / 2:.3f}" if w is not None else "   —   "
            lines.append(
                f"[{color}]{arm:<12}[/{color}] [{cc}]{bar}[/{cc}] [{cc}]{w_s} {label}[/{cc}]"
            )
            if mean is not None:
                lines.append(f"  [#606060]mean={mean:.3f}  n={preds}[/#606060]")
            lines.append("")
        self.query_one("#ci-chart", Static).update("\n".join(lines))

    def _render_phase(self, report: dict, diag: dict) -> None:
        converged = report.get("converged")
        leader    = report.get("leading_arm") or "—"
        step      = self._step

        if step == 0:
            phase, pc = "COLD START", "#606060"
            desc = "No data. Model uses equal priors across all arms."
        elif step < 100:
            phase, pc = "EXPLORATION", "#60A5FA"
            desc = "Gathering first observations. All arms sampled."
        elif converged is None:
            phase, pc = "LEARNING", "#EAB308"
            desc = "Fitting reward model. Confidence intervals still wide."
        elif not converged:
            phase, pc = "FOCUSING", "#F472B6"
            desc = f"Leader: {leader}. CIs overlap — not decisive yet."
        else:
            phase, pc = "CONVERGED", "#22C55E"
            desc = f"Winner: {leader}. Statistical significance reached."

        entropy_s = f"{self._entropy_hist[-1]:.3f}" if self._entropy_hist else "—"
        self.query_one("#phase-panel", Static).update(
            f"[{pc}]◉ {phase}[/{pc}]\n"
            f"[#A0A0A0]{desc}[/#A0A0A0]\n\n"
            f"[#606060]steps {step:,}  entropy {entropy_s}[/#606060]"
        )


# ── main screen ───────────────────────────────────────────────────────────────

class BanditDBDashboard(App):
    TITLE = "BanditDB Monitor"
    CSS   = DASHBOARD_CSS
    BINDINGS = [
        Binding("q", "quit",     "Quit"),
        Binding("r", "refresh",  "Refresh"),
        Binding("s", "simulate", "Simulate"),
    ]

    _overview: reactive[dict] = reactive({})
    _last_refresh: reactive[float] = reactive(0.0)
    _refresh_interval: int = 5

    def __init__(self, db: Client, refresh_interval: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._db = db
        self._refresh_interval = refresh_interval
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._loading = False

    # ── composition ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Horizontal(id="header-bar"):
            yield Static("", id="hbar-content")
        with Horizontal(id="stat-tiles"):
            yield Static("", id="tile-wal",       classes="stat-tile")
            yield Static("", id="tile-campaigns", classes="stat-tile")
            yield Static("", id="tile-preds",     classes="stat-tile")
            yield Static("", id="tile-alerts",    classes="stat-tile")
            yield Static("", id="tile-converged", classes="stat-tile")
        yield Static("CAMPAIGNS  [dim]↑↓ navigate · Enter drill-down · S simulate[/dim]", id="section-campaigns")
        yield DataTable(id="campaign-table")
        yield ScrollableContainer(
            Static("", id="alerts-content"),
            id="alerts-container",
        )
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        tbl: DataTable = self.query_one("#campaign-table", DataTable)
        tbl.add_columns(
            "Campaign ID", "Algo", "Arms", "Predictions",
            "Rewards", "Gap", "Entropy", "Trend", "Convergence", "Leader"
        )
        tbl.cursor_type = "row"
        self.action_refresh()
        self.set_interval(self._refresh_interval, self.action_refresh)

    # ── actions ───────────────────────────────────────────────────────────────

    def action_refresh(self) -> None:
        if self._loading:
            return
        self._loading = True
        self._update_status("Refreshing…")
        future = self._executor.submit(fetch_overview, self._db)
        self.set_timer(0.1, lambda: self._poll_overview(future))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        rows = self._overview.get("campaigns", [])
        row_idx = event.cursor_row
        if row_idx < len(rows):
            cid = rows[row_idx]["id"]
            self.push_screen(CampaignDetailScreen(self._db, cid))

    def action_simulate(self) -> None:
        rows = self._overview.get("campaigns", [])
        tbl: DataTable = self.query_one("#campaign-table", DataTable)
        row_idx = tbl.cursor_row
        if row_idx < len(rows):
            cid = rows[row_idx]["id"]
            self.push_screen(SimulationScreen(self._db, cid))

    def _poll_overview(self, future) -> None:
        if future.done():
            self._loading = False
            self._overview = future.result()
            self._last_refresh = time.time()
            self._render_overview()
        else:
            self.set_timer(0.1, lambda: self._poll_overview(future))

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render_overview(self) -> None:
        data      = self._overview
        error     = data.get("error")
        campaigns = data.get("campaigns", [])
        metrics   = data.get("metrics", {})
        alerts    = data.get("alerts", [])

        if error:
            self._update_status(f"[#EF4444]Error: {error}[/#EF4444]")

        # WAL health
        wal_raw  = metrics.get("banditdb_wal_healthy", "1")
        wal_ok   = wal_raw.strip() == "1"
        wal_text = "[#22C55E]● WAL OK[/#22C55E]" if wal_ok else "[#EF4444]✗ WAL DEGRADED[/#EF4444]"

        # HTTP request rates (predict endpoint)
        preds_total = 0
        for k, v in metrics.items():
            if "predictions" in k or ('arm_predictions_total' in k):
                try:
                    preds_total += int(float(v))
                except Exception:
                    pass

        active_count = len(campaigns)
        alert_count  = len(alerts)
        converged_count = sum(1 for c in campaigns if c.get("converged") is True)

        # Header bar
        ts = time.strftime("%H:%M:%S", time.localtime(self._last_refresh)) if self._last_refresh else "—"
        self.query_one("#hbar-content", Static).update(
            f"[bold #F97316]BanditDB[/bold #F97316]  {wal_text}  "
            f"[#606060]auto-refresh {self._refresh_interval}s · last {ts}[/#606060]"
        )

        # Stat tiles
        self.query_one("#tile-wal", Static).update(
            f"[#606060]WAL[/#606060]\n{wal_text}"
        )
        self.query_one("#tile-campaigns", Static).update(
            f"[#606060]CAMPAIGNS[/#606060]\n[bold #F5F5F5]{active_count}[/bold #F5F5F5] [#606060]active[/#606060]"
        )
        self.query_one("#tile-preds", Static).update(
            f"[#606060]TOTAL PREDS[/#606060]\n[bold #F5F5F5]{_fmt_num(preds_total)}[/bold #F5F5F5]"
        )
        alert_color = "#EF4444" if alert_count else "#22C55E"
        alert_label = f"{alert_count} alert{'s' if alert_count != 1 else ''}" if alert_count else "no alerts"
        self.query_one("#tile-alerts", Static).update(
            f"[#606060]ENTROPY[/#606060]\n[{alert_color}]{alert_label}[/{alert_color}]"
        )
        self.query_one("#tile-converged", Static).update(
            f"[#606060]CONVERGED[/#606060]\n[#22C55E]{converged_count}[/#22C55E][#606060]/{active_count}[/#606060]"
        )

        # Campaign table — save cursor before clear, restore after repopulate
        tbl: DataTable = self.query_one("#campaign-table", DataTable)
        saved_cursor = tbl.cursor_row
        tbl.clear()
        for row in campaigns:
            cid       = row["id"]
            algo      = row["algorithm"]
            arm_count = row["arm_count"]
            preds     = row["total_preds"]
            rewards   = row["total_rewards"]
            e_status  = row["entropy_status"]
            e_trend   = row["entropy_trend"]
            converged = row["converged"]
            leader    = row["leading_arm"]

            algo_c    = _algo_color(algo)
            algo_str  = _algo_name(algo)
            ec, es = _entropy_style(e_status)
            cc, cs = _conv_style(converged)
            trend_color = {
                "falling":   "#EF4444",
                "recovering":"#22C55E",
                "stable":    "#A0A0A0",
                "unknown":   "#606060",
            }.get(e_trend, "#606060")

            tbl.add_row(
                f"[bold]{cid}[/bold]",
                f"[{algo_c}]{algo_str}[/{algo_c}]",
                str(arm_count),
                _fmt_num(preds),
                _fmt_num(rewards),
                _reward_gap_pct(preds, rewards),
                f"[{ec}]{es} {e_status}[/{ec}]",
                f"[{trend_color}]{e_trend}[/{trend_color}]",
                f"[{cc}]{cs}[/{cc}]",
                f"[#A0A0A0]{leader}[/#A0A0A0]",
            )

        if saved_cursor is not None and saved_cursor < len(campaigns):
            tbl.move_cursor(row=saved_cursor)

        # Alerts panel
        if alerts:
            lines = []
            for a in alerts:
                status = a["status"]
                color  = "#EF4444" if status == "critical" else "#EAB308"
                symbol = "✗" if status == "critical" else "⚠"
                lines.append(
                    f"[{color}]{symbol} {a['campaign']}[/{color}]"
                    f"[#606060] — {a['cause']}[/#606060]"
                    + (f"  [#F97316]→ {a['action']}[/#F97316]" if a.get("action") else "")
                )
            self.query_one("#alerts-content", Static).update("\n".join(lines))
        else:
            self.query_one("#alerts-content", Static).update(
                "[#22C55E]✓ All campaigns healthy[/#22C55E]"
            )

        self._update_status(
            f"[#606060]{active_count} campaigns · {alert_count} alerts · "
            f"{converged_count} converged · refreshed {ts}[/#606060]"
        )

    def _update_status(self, msg: str) -> None:
        try:
            self.query_one("#status-bar", Static).update(msg)
        except Exception:
            pass


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="banditdb-dashboard",
        description="Terminal dashboard for BanditDB.",
    )
    parser.add_argument(
        "--url", "-u",
        default=os.environ.get("BANDITDB_URL", "http://localhost:8080"),
        help="BanditDB server URL (default: $BANDITDB_URL or http://localhost:8080)",
    )
    parser.add_argument(
        "--api-key", "-k",
        default=os.environ.get("BANDITDB_API_KEY"),
        help="API key (default: $BANDITDB_API_KEY)",
    )
    parser.add_argument(
        "--refresh", "-n",
        type=int,
        default=int(os.environ.get("BANDITDB_DASHBOARD_REFRESH", "5")),
        help="Auto-refresh interval in seconds (default: 5)",
    )
    args = parser.parse_args()

    # max_retries=0: fail fast — retrying timed-out calls floods the server with CLOSE_WAIT connections
    db  = Client(url=args.url, api_key=args.api_key, max_retries=0)
    app = BanditDBDashboard(db=db, refresh_interval=args.refresh)
    app.run()


if __name__ == "__main__":
    main()
