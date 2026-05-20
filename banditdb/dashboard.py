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
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

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

def _algo_color(algo: str) -> str:
    colors = {
        "linucb":      "#60A5FA",
        "thompson":    "#A78BFA",
        "neural":      "#F472B6",
        "tournament":  "#FB923C",
    }
    return colors.get(algo.lower(), "#A0A0A0")

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
        hdr = self.query_one("#detail-header", Static)
        hdr.update(
            f"← [dim]Back (Esc)[/dim]  [bold #F5F5F5]{self._campaign_id}[/bold #F5F5F5]  "
            f"[{algo_color}]{algo}[/{algo_color}] · α={alpha} · {arm_count} arms"
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


# ── main screen ───────────────────────────────────────────────────────────────

class BanditDBDashboard(App):
    TITLE = "BanditDB Monitor"
    CSS   = DASHBOARD_CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
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
        yield Static("CAMPAIGNS  [dim]↑↓ navigate · Enter drill-down[/dim]", id="section-campaigns")
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

            algo_c = _algo_color(algo)
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
                f"[{algo_c}]{algo}[/{algo_c}]",
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

    db  = Client(url=args.url, api_key=args.api_key)
    app = BanditDBDashboard(db=db, refresh_interval=args.refresh)
    app.run()


if __name__ == "__main__":
    main()
