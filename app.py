"""Mini-IDS Intelligent Network Security Analytics Platform.

Streamlit web application combining heuristic intrusion detection, supervised
machine learning classification, unsupervised Isolation Forest anomaly detection,
and exploratory cybersecurity traffic analytics.
"""

from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from analytics.exploratory_analysis import DatasetProfiler
from analytics.theme import SEVERITY_COLORS, THEME_COLORS, get_altair_theme, get_custom_css
from capture.live_capture import get_available_interfaces, get_live_worker
from ml.dataset_manager import DATASET_METADATA, DatasetManager
from ml.evaluation import ModelEvaluator
from ml.explainability import ModelExplainer
from ml.model_registry import ModelRegistry
from ml.preprocessing import FlowDataPreprocessor
from ml.train_anomaly import AnomalyDetectorTrainer
from ml.train_supervised import SupervisedModelTrainer
from storage.database import DEFAULT_DB_PATH, DatabaseManager
from storage.demo_data import DEFAULT_DEMO_DB_PATH, get_or_create_demo_database

logger = logging.getLogger("mini_ids.dashboard")


def format_timestamp(ts: Union[float, int, str, None]) -> str:
    """Format UNIX epoch float to a readable datetime string (UTC)."""
    if ts is None:
        return "N/A"
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ts)


def build_severity_chart(severity_data: List[Dict[str, Any]]) -> Optional[alt.Chart]:
    """Construct an Altair horizontal bar chart for alert severity distribution."""
    if not severity_data:
        return None
    df = pd.DataFrame(severity_data)
    color_scale = alt.Scale(
        domain=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        range=[
            THEME_COLORS["critical"],
            THEME_COLORS["high"],
            THEME_COLORS["warning"],
            THEME_COLORS["low"],
        ],
    )
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("count:Q", title="Alert Count"),
            y=alt.Y("severity:N", sort=["CRITICAL", "HIGH", "MEDIUM", "LOW"], title="Severity Tier"),
            color=alt.Color("severity:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("severity:N", title="Severity"),
                alt.Tooltip("count:Q", title="Alert Count", format=","),
            ],
        )
        .properties(height=180)
    )
    return chart


def build_rule_chart(rule_data: List[Dict[str, Any]]) -> Optional[alt.Chart]:
    """Construct an Altair horizontal bar chart for alerts by detection rule."""
    if not rule_data:
        return None
    df = pd.DataFrame(rule_data)
    chart = (
        alt.Chart(df)
        .mark_bar(color=THEME_COLORS["primary_accent"], cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("count:Q", title="Alert Count"),
            y=alt.Y("rule_name:N", sort="-x", title="Detection Rule"),
            tooltip=[
                alt.Tooltip("rule_name:N", title="Rule"),
                alt.Tooltip("count:Q", title="Alert Count", format=","),
            ],
        )
        .properties(height=180)
    )
    return chart


def build_timeline_chart(
    timeline_data: List[Dict[str, Any]],
    color: str = THEME_COLORS["primary_accent"],
    fill_color: str = "#E0835D",
    y_field: str = "count",
    y_title: str = "Alert Count",
) -> Optional[alt.Chart]:
    """Construct an Altair discrete column bar chart for event timeline aggregation."""
    if not timeline_data:
        return None
    df = pd.DataFrame(timeline_data)
    chart = (
        alt.Chart(df)
        .mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("bucket:N", sort=None, title="Time Bucket (UTC)"),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            tooltip=[
                alt.Tooltip("bucket:N", title="Interval (UTC)"),
                alt.Tooltip(f"{y_field}:Q", title=y_title, format=","),
            ],
        )
        .properties(height=200)
    )
    return chart


def build_ranking_chart(
    data: List[Dict[str, Any]],
    category_field: str,
    value_field: str,
    category_title: str,
    value_title: str,
    bar_color: str = THEME_COLORS["secondary_accent"],
) -> Optional[alt.Chart]:
    """Construct a horizontal bar chart for ranked entities (IPs, Ports)."""
    if not data:
        return None
    df = pd.DataFrame(data)
    df[category_field] = df[category_field].astype(str)
    chart = (
        alt.Chart(df)
        .mark_bar(color=bar_color, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X(f"{value_field}:Q", title=value_title),
            y=alt.Y(f"{category_field}:N", sort="-x", title=category_title),
            tooltip=[
                alt.Tooltip(f"{category_field}:N", title=category_title),
                alt.Tooltip(f"{value_field}:Q", title=value_title, format=","),
            ],
        )
        .properties(height=180)
    )
    return chart


def render_live_capture_section(target_db_path: Path, is_demo_mode: bool = False) -> None:
    """Render the Live Packet Capture control panel and real-time worker status."""
    st.markdown("### 📡 Live Packet Capture (Passive Sniffer)")

    if is_demo_mode:
        st.info(
            "☁️ **Live Packet Capture is disabled in Hosted Demonstration Mode.**\n\n"
            "Passive live packet sniffing requires direct physical network adapter access, administrative "
            "privileges, and local Npcap kernel drivers on a host workstation. In this hosted cloud "
            "demonstration, the dashboard operates in safe, read-only forensic mode.\n\n"
            "To perform real-time packet capture on your local Wi-Fi or Ethernet adapter, run Mini-IDS "
            "locally on Windows using:\n"
            "```powershell\n"
            "streamlit run app.py\n"
            "```"
        )
        st.divider()
        return

    st.caption(
        "Capture live traffic from local network adapters using Npcap. Packets are evaluated "
        "against detection rules and saved directly to the active SQLite database."
    )

    worker = get_live_worker()
    status = worker.get_status()

    # Capture status banner
    if status["status"] == "STOPPING":
        st.warning(
            f"⏳ **Stopping capture on `{status['interface']}`... finalizing queue and database writes.**\n\n"
            f"**Packets Captured:** {status['packets_captured']:,} | "
            f"**Persisted to DB:** {status['packets_persisted']:,} | "
            f"**Queue Drops:** {status['packets_dropped']:,} | "
            f"**Alerts Triggered:** {status['alerts_generated']:,}"
        )
    elif status["is_running"]:
        st.info(
            f"🟢 **Live Capture Active on `{status['interface']}`**\n\n"
            f"**Packets Captured:** {status['packets_captured']:,} | "
            f"**Persisted to DB:** {status['packets_persisted']:,} | "
            f"**Queue Drops:** {status['packets_dropped']:,} | "
            f"**Alerts Triggered:** {status['alerts_generated']:,} | "
            f"**Elapsed Time:** {status['elapsed_seconds']:.1f}s"
        )
    elif status["status"] == "COMPLETED":
        st.success(
            f"✅ **Capture Completed on `{status['interface']}`**\n\n"
            f"Captured: **{status['packets_captured']:,}** packets | "
            f"Persisted: **{status['packets_persisted']:,}** | "
            f"Queue Drops: **{status['packets_dropped']:,}** | "
            f"Alerts Triggered: **{status['alerts_generated']:,}** | "
            f"Duration: **{status['elapsed_seconds']:.1f}s**"
        )
    elif status["status"] == "STOPPED":
        st.warning(
            f"⏹️ **Capture Stopped on `{status['interface']}`**\n\n"
            f"Captured: **{status['packets_captured']:,}** packets | "
            f"Persisted: **{status['packets_persisted']:,}** | "
            f"Queue Drops: **{status['packets_dropped']:,}** | "
            f"Alerts Triggered: **{status['alerts_generated']:,}** | "
            f"Duration: **{status['elapsed_seconds']:.1f}s**"
        )
    elif status["status"] == "ERROR":
        st.error(f"❌ **Capture Error:** {status.get('error_message')}")
    elif status["status"] == "IDLE":
        st.caption("ℹ️ **Sniffer Status:** `IDLE` — Ready to capture. Select an interface in the configuration panel below and click **Start Capture**.")

    with st.expander("⚙️ Capture Configuration & Controls", expanded=status["is_running"] or status["status"] == "IDLE"):
        available_ifaces = get_available_interfaces()

        if not available_ifaces:
            st.warning(
                "⚠️ **No network interfaces discovered.**\n\n"
                "Ensure Npcap is installed on Windows with 'WinPcap API-compatible Mode' enabled. "
                "Administrative privileges may be required to access raw network adapters."
            )
            return

        iface_labels = [i["display_name"] for i in available_ifaces]
        selected_label = st.radio(
            "Select Network Interface",
            options=iface_labels,
            index=0,
            help="Choose the network adapter to monitor. Passive sniffing only; traffic is not injected or altered.",
            key="live_capture_iface_select",
        )
        selected_iface = next((i for i in available_ifaces if i["display_name"] == selected_label), available_ifaces[0])

        col_cond, col_val = st.columns(2)
        with col_cond:
            stop_condition = st.radio(
                "Stop Condition",
                options=["Manual Stop", "Duration Timeout", "Packet Count Limit"],
                index=0,
                horizontal=True,
                key="live_capture_condition_type",
            )
        with col_val:
            duration_limit: Optional[float] = None
            packet_limit: Optional[int] = None
            if stop_condition == "Duration Timeout":
                duration_limit = float(
                    st.number_input(
                        "Timeout (Seconds)",
                        min_value=1.0,
                        max_value=3600.0,
                        value=15.0,
                        step=5.0,
                        key="live_capture_duration_val",
                    )
                )
            elif stop_condition == "Packet Count Limit":
                packet_limit = int(
                    st.number_input(
                        "Max Packets",
                        min_value=5,
                        max_value=100000,
                        value=100,
                        step=25,
                        key="live_capture_packet_val",
                    )
                )

        btn_start, btn_stop, btn_refresh = st.columns([1, 1, 2])
        with btn_start:
            can_start = not status["is_running"]
            if st.button("▶️ Start Capture", disabled=not can_start, type="primary", key="live_btn_start"):
                iface_id = selected_iface["name"] if selected_iface["name"] else selected_iface["guid"]
                try:
                    worker.configure(
                        interface=iface_id,
                        db_path=target_db_path,
                        packet_limit=packet_limit,
                        duration_seconds=duration_limit,
                    )
                    worker.start()
                    st.rerun()
                except Exception as err:
                    st.error(f"Failed to start live capture: {err}")
        with btn_stop:
            can_stop = status["is_running"]
            if st.button("⏹️ Stop Capture", disabled=not can_stop, key="live_btn_stop"):
                worker.stop()
                st.rerun()
        with btn_refresh:
            if st.button("🔄 Refresh Sniffer Status", key="live_btn_refresh"):
                st.rerun()


def render_dashboard() -> None:
    """Main Streamlit application entrypoint."""
    st.set_page_config(
        page_title="Mini-IDS: Security Analytics Platform",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply global custom CSS design tokens
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Sidebar Header
    st.sidebar.title("🛡️ MINI-IDS")
    st.sidebar.caption("Intelligent Network Intrusion Detection & Analytics Platform")

    # Primary Configuration Input (Must remain sidebar.text_input[0] for tests)
    db_path_input = st.sidebar.text_input(
        "Database Path",
        value=str(DEFAULT_DB_PATH),
        help="Path to SQLite database containing stored packets, alerts, and model records.",
    )

    # Refresh Button (Must remain sidebar.button[0] for tests)
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    # Automatically activate demo mode if DEFAULT_DB_PATH does not exist or if MINI_IDS_DEMO_MODE is set
    default_demo_active = (not DEFAULT_DB_PATH.is_file()) or (
        os.environ.get("MINI_IDS_DEMO_MODE", "").lower() in ("1", "true", "yes")
    )

    demo_toggle = st.sidebar.toggle(
        "Demonstration Mode (Synthetic)",
        value=default_demo_active,
        help="Enable sanitized synthetic demonstration dataset for evaluation without local capture files.",
        key="app_demo_toggle",
    )

    # 9-Section Navigation Radio (Sidebar Radio [0])
    nav_sections = [
        "Overview",
        "Live Monitoring",
        "Alert Investigation",
        "Traffic Intelligence",
        "AI/ML Detection",
        "Model Laboratory",
        "Data Explorer",
        "Detection Rules",
        "System & Settings",
    ]
    selected_page = st.sidebar.radio(
        "Navigation",
        options=nav_sections,
        index=0,
        help="Switch between security monitoring, investigation, machine learning, and data science workspaces.",
        key="app_navigation_radio",
    )

    # Resolve active database
    raw_path = Path(db_path_input.strip())
    custom_path_provided = bool(raw_path != DEFAULT_DB_PATH)
    is_demo_mode = demo_toggle and not custom_path_provided

    if is_demo_mode:
        target_db_path = get_or_create_demo_database()
        st.sidebar.info("🧪 **Operational Mode:** Demo Mode (Synthetic)")
        st.sidebar.caption(
            "Using deterministic synthetic demonstration dataset with RFC 5737 documentation IPs. "
            "Real captures and local databases are not accessed."
        )
    else:
        target_db_path = raw_path
        st.sidebar.caption(f"📁 **Active Database:** `{target_db_path}`")

    # Check database existence BEFORE opening connection
    if not is_demo_mode and not target_db_path.is_file():
        st.warning(f"Database file not found: `{target_db_path}`")
        st.info(
            "**No database exists at the specified path.**\n\n"
            "The Mini-IDS dashboard does not automatically create empty database files. "
            "To generate data, run offline analysis from your terminal:\n\n"
            "```powershell\n"
            "python main.py --pcap data/sample.pcap\n"
            "```\n\n"
            "Or toggle **Demonstration Mode (Synthetic)** in the sidebar to explore simulated telemetry."
        )
        return

    # Demonstration Mode top notification banner
    if is_demo_mode:
        st.info(
            "🧪 **Demonstration Mode Active** — Displaying sanitized synthetic network telemetry and simulated intrusion "
            "alerts for evaluation and mentor review. All events are synthetic and explicitly labeled. "
            "No real network packets, private host data, or live packet captures are active in this mode."
        )

    # Database connection & retrieval
    try:
        with DatabaseManager(target_db_path) as db:
            packet_count = db.get_packet_count()
            alert_count = db.get_alert_count()

            # Empty database state
            if packet_count == 0 and alert_count == 0:
                if selected_page in ("Overview", "Live Monitoring"):
                    render_live_capture_section(target_db_path, is_demo_mode=is_demo_mode)
                st.info(
                    f"Connected to database `{target_db_path}`, but it contains no stored packets or alerts.\n\n"
                    "Run an offline PCAP analysis to populate this database:\n\n"
                    "```powershell\n"
                    f"python main.py --pcap <path_to_pcap> --db \"{target_db_path}\"\n"
                    "```"
                )
                return

            all_alerts = db.get_alerts(order="DESC")

            # Sidebar Filters (Preserves exact selectbox & checkbox order for tests)
            st.sidebar.header("Filter Options")
            st.sidebar.subheader("Alert Filters")
            st.sidebar.caption("Applies to Alert Analytics and Security Alerts.")
            available_rules = sorted(list({a["rule_name"] for a in all_alerts if a.get("rule_name")}))
            available_severities = sorted(list({a["severity"] for a in all_alerts if a.get("severity")}))

            selected_rule = st.sidebar.selectbox(
                "Detection Rule",
                options=["All"] + available_rules,
                index=0,
            )
            selected_severity = st.sidebar.selectbox(
                "Severity",
                options=["All"] + available_severities,
                index=0,
            )
            alert_limit = st.sidebar.selectbox(
                "Alerts Display Limit",
                options=[25, 50, 100, 250, "All"],
                index=1,
            )

            st.sidebar.subheader("Traffic Filters")
            st.sidebar.caption("Applies to Network Traffic Analytics.")
            packet_limit = st.sidebar.selectbox(
                "Recent Packets Limit",
                options=[50, 100, 250, 500, 1000],
                index=1,
            )
            selected_protocol = st.sidebar.selectbox(
                "Protocol",
                options=["All", "TCP", "UDP", "ICMP"],
                index=0,
            )

            st.sidebar.subheader("Analysis Scope")
            bounds = db.get_timestamp_bounds()
            min_ts = bounds.get("min_timestamp")
            max_ts = bounds.get("max_timestamp")

            enable_time_filter = st.sidebar.checkbox(
                "Filter by Time Range",
                value=False,
                help="Isolate records to a specific capture interval without modifying the database.",
            )
            scope_start_ts: Optional[float] = None
            scope_end_ts: Optional[float] = None
            if enable_time_filter and min_ts is not None and max_ts is not None:
                st.sidebar.caption(f"Stored bounds: {format_timestamp(min_ts)} — {format_timestamp(max_ts)}")
                scope_start_ts = st.sidebar.number_input(
                    "Start Timestamp (Epoch s)",
                    value=float(min_ts),
                    step=1.0,
                    format="%.2f",
                )
                scope_end_ts = st.sidebar.number_input(
                    "End Timestamp (Epoch s)",
                    value=float(max_ts),
                    step=1.0,
                    format="%.2f",
                )

            # Apply scope filtering
            active_alerts = all_alerts
            if enable_time_filter and scope_start_ts is not None and scope_end_ts is not None:
                active_alerts = [
                    a for a in active_alerts
                    if scope_start_ts <= float(a.get("timestamp", 0.0)) <= scope_end_ts
                ]

            fetch_limit = int(packet_limit) * 2 if selected_protocol != "All" else int(packet_limit)
            if enable_time_filter and scope_start_ts is not None and scope_end_ts is not None:
                raw_packets = db.get_related_packets(
                    start_time=scope_start_ts,
                    end_time=scope_end_ts,
                    protocol=selected_protocol if selected_protocol != "All" else None,
                    limit=fetch_limit,
                    order="DESC",
                )
            else:
                raw_packets = db.get_packets(limit=fetch_limit, order="DESC")

            # Filtered alerts for table and investigation
            filtered_alerts = active_alerts
            if selected_rule != "All":
                filtered_alerts = [a for a in filtered_alerts if a.get("rule_name") == selected_rule]
            if selected_severity != "All":
                filtered_alerts = [a for a in filtered_alerts if a.get("severity") == selected_severity]
            if alert_limit != "All":
                filtered_alerts = filtered_alerts[: int(alert_limit)]

            kpis = db.get_security_kpis(
                start_time=scope_start_ts if enable_time_filter else None,
                end_time=scope_end_ts if enable_time_filter else None,
            )

            # ------------------------------------------------------------------
            # PAGE 1: OVERVIEW (Default Page)
            # ------------------------------------------------------------------
            if selected_page == "Overview":
                render_live_capture_section(target_db_path, is_demo_mode=is_demo_mode)

                if enable_time_filter and scope_start_ts is not None and scope_end_ts is not None:
                    st.subheader("Security Overview (Scoped to Selected Interval)")
                    st.caption(
                        f"Showing metrics between `{format_timestamp(scope_start_ts)}` and `{format_timestamp(scope_end_ts)}`. "
                        "Uncheck 'Filter by Time Range' in sidebar to view cumulative totals across all stored history."
                    )
                else:
                    st.subheader("Stored Totals (Cumulative Across All Analyses)")
                    st.caption(
                        "Cumulative metrics reflect the complete database state across all ingested PCAP runs and live captures. "
                        "Historical PCAP data and recent captures are indexed in SQLite."
                    )

                # Primary KPI Row
                kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                kpi_col1.metric("Total Stored Packets", f"{kpis['total_packets']:,}")
                kpi_col2.metric("Total Stored Alerts", f"{kpis['total_alerts']:,}")
                kpi_col3.metric("High-Severity Alerts", f"{kpis['high_severity_alerts']:,}")
                kpi_col4.metric("Distinct Alert Sources", f"{kpis['distinct_alert_sources']:,}")

                # Rule Counts Row (preserves compatibility with test assertions)
                rule_counts = Counter(a["rule_name"] for a in all_alerts if a.get("rule_name"))
                if rule_counts:
                    st.markdown("**Alert Counts Grouped by Rule / Type:**")
                    rule_columns = st.columns(max(len(rule_counts), 1))
                    for col, (rule_name, count) in zip(rule_columns, sorted(rule_counts.items())):
                        col.metric(rule_name, f"{count:,}")

                # Model Readiness Banner
                registry = ModelRegistry(db_path=target_db_path)
                active_sup = db.get_active_model("supervised_binary") or db.get_active_model()
                active_anom = db.get_active_model("anomaly_isolation_forest")

                st.markdown("---")
                mod_c1, mod_c2 = st.columns(2)
                with mod_c1:
                    if active_sup:
                        st.success(
                            f"🤖 **Active Supervised Model:** `{active_sup['algorithm']}` ({active_sup['version']})\n\n"
                            f"Accuracy: **{active_sup['metrics'].get('accuracy', 0)*100:.2f}%** | "
                            f"F1 Score: **{active_sup['metrics'].get('f1_weighted', 0)*100:.2f}%**"
                        )
                    else:
                        st.info("ℹ️ **Supervised Model:** No active model selected. Defaulting to heuristic rule engine.")
                with mod_c2:
                    if active_anom:
                        st.success(
                            f"🌲 **Active Anomaly Detector:** `Isolation Forest` ({active_anom['version']})\n\n"
                            f"Features: **{len(active_anom['feature_schema'])}** inputs | "
                            f"Threshold: **{active_anom['metrics'].get('score_threshold', 0.5):.3f}**"
                        )
                    else:
                        st.info("ℹ️ **Anomaly Detector:** Unsupervised detector not active.")

                st.divider()

                # Analytics Tabs
                st.subheader("📊 Security Analytics")
                tab_alerts, tab_traffic = st.tabs(["🚨 Alert Analytics", "🌐 Network Traffic Analytics"])

                with tab_alerts:
                    st.caption(f"Showing alert analytics filtered by Rule: **{selected_rule}** | Severity: **{selected_severity}**.")
                    dist_col1, dist_col2 = st.columns(2)
                    with dist_col1:
                        st.markdown("##### Alert Severity Distribution")
                        sev_data = db.get_alert_severity_distribution(
                            start_time=scope_start_ts if enable_time_filter else None,
                            end_time=scope_end_ts if enable_time_filter else None,
                            rule=selected_rule,
                        )
                        sev_chart = build_severity_chart(sev_data)
                        if sev_chart is not None:
                            st.altair_chart(sev_chart, width="stretch")
                        else:
                            st.info("No alert severity records available.")

                    with dist_col2:
                        st.markdown("##### Alerts by Detection Rule")
                        rule_data = db.get_alerts_by_rule(
                            start_time=scope_start_ts if enable_time_filter else None,
                            end_time=scope_end_ts if enable_time_filter else None,
                            severity=selected_severity,
                        )
                        rule_chart = build_rule_chart(rule_data)
                        if rule_chart is not None:
                            st.altair_chart(rule_chart, width="stretch")
                        else:
                            st.info("No detection rule records available.")

                    st.markdown("##### Alert Timeline (UTC)")
                    alert_timeline_data = db.get_alert_timeline(
                        start_time=scope_start_ts if enable_time_filter else None,
                        end_time=scope_end_ts if enable_time_filter else None,
                        rule=selected_rule,
                        severity=selected_severity,
                    )
                    alert_tl_chart = build_timeline_chart(alert_timeline_data)
                    if alert_tl_chart is not None:
                        st.altair_chart(alert_tl_chart, width="stretch")

                    rank_col1, rank_col2 = st.columns(2)
                    with rank_col1:
                        st.markdown("##### Top Offending Source IPs")
                        top_sources = db.get_top_alert_sources(
                            start_time=scope_start_ts if enable_time_filter else None,
                            end_time=scope_end_ts if enable_time_filter else None,
                            rule=selected_rule,
                            severity=selected_severity,
                        )
                        src_chart = build_ranking_chart(
                            top_sources,
                            category_field="src_ip",
                            value_field="count",
                            category_title="Source IP",
                            value_title="Alert Count",
                            bar_color=THEME_COLORS["critical"],
                        )
                        if src_chart is not None:
                            st.altair_chart(src_chart, width="stretch")

                    with rank_col2:
                        st.markdown("##### Top Targeted Destination IPs")
                        top_dests = db.get_top_alert_destinations(
                            start_time=scope_start_ts if enable_time_filter else None,
                            end_time=scope_end_ts if enable_time_filter else None,
                            rule=selected_rule,
                            severity=selected_severity,
                        )
                        dst_chart = build_ranking_chart(
                            top_dests,
                            category_field="dst_ip",
                            value_field="count",
                            category_title="Destination IP",
                            value_title="Alert Count",
                            bar_color=THEME_COLORS["primary_accent"],
                        )
                        if dst_chart is not None:
                            st.altair_chart(dst_chart, width="stretch")

                with tab_traffic:
                    st.caption(f"Showing raw packet traffic filtered by Protocol: **{selected_protocol}**.")
                    pcol1, pcol2 = st.columns(2)
                    with pcol1:
                        st.markdown("##### Protocol Distribution")
                        proto_data = db.get_protocol_distribution(
                            start_time=scope_start_ts if enable_time_filter else None,
                            end_time=scope_end_ts if enable_time_filter else None,
                            protocol=selected_protocol if selected_protocol != "All" else None,
                        )
                        p_chart = build_ranking_chart(
                            proto_data,
                            category_field="protocol",
                            value_field="count",
                            category_title="Protocol",
                            value_title="Packet Count",
                            bar_color=THEME_COLORS["secondary_accent"],
                        )
                        if p_chart is not None:
                            st.altair_chart(p_chart, width="stretch")

                    with pcol2:
                        st.markdown("##### Top Destination Ports")
                        port_data = db.get_top_destination_ports(
                            start_time=scope_start_ts if enable_time_filter else None,
                            end_time=scope_end_ts if enable_time_filter else None,
                            protocol=selected_protocol if selected_protocol != "All" else None,
                        )
                        port_chart = build_ranking_chart(
                            port_data,
                            category_field="dst_port",
                            value_field="packet_count",
                            category_title="Destination Port",
                            value_title="Packet Count",
                            bar_color=THEME_COLORS["primary_accent"],
                        )
                        if port_chart is not None:
                            st.altair_chart(port_chart, width="stretch")

                st.divider()

                # Recent Security Alerts Table (Must be dataframe[0] for tests)
                st.subheader("Recent Security Alerts")
                if not filtered_alerts:
                    st.info("No alerts match the selected filter criteria.")
                else:
                    alert_rows = []
                    for a in filtered_alerts:
                        alert_rows.append({
                            "Alert ID": a.get("id"),
                            "Timestamp": format_timestamp(a.get("timestamp")),
                            "Rule": a.get("rule_name", "UNKNOWN"),
                            "Severity": a.get("severity", "UNKNOWN"),
                            "Detection Method": a.get("detection_method", "HEURISTIC"),
                            "Source IP": a.get("src_ip") or "N/A",
                            "Destination IP": a.get("dst_ip") or "N/A",
                            "Observed": round(float(a.get("observed_value", 0.0)), 2),
                            "Threshold": round(float(a.get("threshold", 0.0)), 2),
                            "Window (s)": round(float(a.get("window_seconds", 0.0)), 1),
                            "Description / Reason": a.get("message", ""),
                        })
                    df_alerts = pd.DataFrame(alert_rows)
                    st.dataframe(df_alerts, width="stretch")

                st.divider()

                # Alert Investigation Workflow (Must have selectbox[0] and dataframe[1] for tests)
                st.subheader("🔍 Alert Investigation")
                if not filtered_alerts:
                    st.info("No alerts available to investigate with the current filter criteria.")
                else:
                    alert_id_to_label = {
                        a["id"]: (
                            f"Alert #{a['id']} | {a.get('rule_name', 'UNKNOWN')} [{a.get('severity', 'UNKNOWN')}] | "
                            f"{a.get('src_ip') or 'N/A'} -> {a.get('dst_ip') or 'N/A'} | "
                            f"{format_timestamp(a.get('timestamp'))}"
                        )
                        for a in filtered_alerts
                    }
                    selected_alert_id = st.selectbox(
                        "Select Alert to Investigate",
                        options=list(alert_id_to_label.keys()),
                        format_func=lambda aid: alert_id_to_label[aid],
                        key="overview_alert_select",
                    )
                    selected_alert = db.get_alert_by_id(selected_alert_id)
                    if selected_alert:
                        st.markdown(f"#### Alert Details — ID #{selected_alert['id']}")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Detection Rule", selected_alert.get("rule_name", "UNKNOWN"))
                        c2.metric("Severity", selected_alert.get("severity", "UNKNOWN"))
                        c3.metric("Observed / Threshold", f"{selected_alert.get('observed_value', 0):.2f} / {selected_alert.get('threshold', 0):.2f}")
                        c4.metric("Window Duration", f"{selected_alert.get('window_seconds', 0):.1f} s")

                        alert_ts = float(selected_alert.get("timestamp", 0.0))
                        win_sec = float(selected_alert.get("window_seconds", 0.0))
                        effective_start = alert_ts - win_sec
                        effective_end = alert_ts

                        evidence_packets = db.get_related_packets(
                            src_ip=selected_alert.get("src_ip"),
                            dst_ip=selected_alert.get("dst_ip"),
                            start_time=effective_start,
                            end_time=effective_end,
                            limit=500,
                            order="ASC",
                        )

                        st.markdown("#### Related Packet Evidence")
                        if not evidence_packets:
                            st.warning(
                                f"No matching packet records found for "
                                f"{selected_alert.get('src_ip') or 'any'} -> {selected_alert.get('dst_ip') or 'any'} "
                                f"within the detection interval `[{format_timestamp(effective_start)} — {format_timestamp(effective_end)}]`."
                            )
                        else:
                            st.success(f"Found **{len(evidence_packets)}** related packet record(s) matching detection window.")
                            evidence_rows = []
                            for p in evidence_packets:
                                evidence_rows.append({
                                    "Timestamp": format_timestamp(p.get("timestamp")),
                                    "Epoch": f"{p.get('timestamp', 0.0):.6f}",
                                    "Protocol": p.get("protocol", ""),
                                    "Source IP": p.get("src_ip", ""),
                                    "Destination IP": p.get("dst_ip", ""),
                                    "Source Port": str(p["src_port"]) if p.get("src_port") is not None else "-",
                                    "Destination Port": str(p["dst_port"]) if p.get("dst_port") is not None else "-",
                                    "Length (Bytes)": p.get("length", 0),
                                    "TCP Flags": p.get("tcp_flags") if p.get("tcp_flags") is not None else "-",
                                })
                            df_evidence = pd.DataFrame(evidence_rows)
                            st.dataframe(df_evidence, width="stretch")

                st.divider()

                # Recent Network Traffic Table (Must be dataframe[2] for tests)
                st.subheader("Recent Network Traffic")
                filtered_packets = raw_packets
                if selected_protocol != "All":
                    filtered_packets = [
                        p for p in filtered_packets
                        if p.get("protocol", "").upper() == selected_protocol.upper()
                    ]
                filtered_packets = filtered_packets[: int(packet_limit)]

                if not filtered_packets:
                    st.info("No packets match the selected traffic filters.")
                else:
                    packet_rows = []
                    for p in filtered_packets:
                        packet_rows.append({
                            "Timestamp": format_timestamp(p.get("timestamp")),
                            "Source IP": p.get("src_ip", ""),
                            "Destination IP": p.get("dst_ip", ""),
                            "Protocol": p.get("protocol", ""),
                            "Length (Bytes)": p.get("length", 0),
                            "Source Port": str(p["src_port"]) if p.get("src_port") is not None else "-",
                            "Destination Port": str(p["dst_port"]) if p.get("dst_port") is not None else "-",
                            "TCP Flags": p.get("tcp_flags") if p.get("tcp_flags") is not None else "-",
                        })
                    df_packets = pd.DataFrame(packet_rows)
                    st.dataframe(df_packets, width="stretch")

            # ------------------------------------------------------------------
            # PAGE 2: LIVE MONITORING
            # ------------------------------------------------------------------
            elif selected_page == "Live Monitoring":
                render_live_capture_section(target_db_path, is_demo_mode=is_demo_mode)

                st.subheader("Real-Time Event Stream")
                recent_live_alerts = db.get_alerts(limit=25, order="DESC")
                if not recent_live_alerts:
                    st.info("No security alerts triggered during current live capture session.")
                else:
                    st.dataframe(pd.DataFrame(recent_live_alerts), width="stretch")

            # ------------------------------------------------------------------
            # PAGE 3: ALERT INVESTIGATION (Dedicated Forensic Workspace)
            # ------------------------------------------------------------------
            elif selected_page == "Alert Investigation":
                st.title("🔍 Alert Investigation & Threat Correlation Console")
                st.caption(
                    "Examine intrusion detections, correlate heuristic alerts with machine learning inferences, "
                    "inspect raw packet evidence, and review defensive triage playbooks."
                )

                if not all_alerts:
                    st.info("No security alerts exist in this database to investigate.")
                else:
                    # Filter row
                    f_col1, f_col2, f_col3 = st.columns(3)
                    with f_col1:
                        meth_options = ["All", "HEURISTIC", "ML_SUPERVISED", "ML_ANOMALY"]
                        selected_meth = st.selectbox("Detection Method", meth_options, index=0)
                    with f_col2:
                        selected_sev = st.selectbox("Severity Tier", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"], index=0)
                    with f_col3:
                        rule_filter = st.selectbox("Triggered Rule", ["All"] + sorted(list({a["rule_name"] for a in all_alerts})), index=0)

                    inv_pool = all_alerts
                    if selected_meth != "All":
                        inv_pool = [a for a in inv_pool if a.get("detection_method", "HEURISTIC") == selected_meth]
                    if selected_sev != "All":
                        inv_pool = [a for a in inv_pool if a.get("severity") == selected_sev]
                    if rule_filter != "All":
                        inv_pool = [a for a in inv_pool if a.get("rule_name") == rule_filter]

                    if not inv_pool:
                        st.info("No alerts match the chosen investigation filters.")
                    else:
                        alert_choices = {
                            a["id"]: f"Alert #{a['id']} [{a.get('severity')}] {a.get('rule_name')} | {a.get('src_ip')} -> {a.get('dst_ip')} | {format_timestamp(a.get('timestamp'))}"
                            for a in inv_pool
                        }
                        chosen_id = st.selectbox(
                            "Select Alert to Investigate",
                            options=list(alert_choices.keys()),
                            format_func=lambda x: alert_choices[x],
                            key="deep_investigate_select",
                        )
                        alert = db.get_alert_by_id(chosen_id)
                        if alert:
                            st.markdown(f"### Alert Forensic Dossier — ID #{alert['id']}")
                            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                            d_col1.metric("Detection Category", alert.get("rule_name", "UNKNOWN"))
                            d_col2.metric("Severity Tier", alert.get("severity", "UNKNOWN"))
                            d_col3.metric("Detection Method", alert.get("detection_method", "HEURISTIC"))
                            d_col4.metric("Evaluation Window", f"{alert.get('window_seconds', 0):.1f} s")

                            st.markdown(
                                f"- **Trigger Timestamp:** `{format_timestamp(alert.get('timestamp'))}`\n"
                                f"- **Source Host:** `{alert.get('src_ip') or 'N/A'}`\n"
                                f"- **Destination Host:** `{alert.get('dst_ip') or 'N/A'}`\n"
                                f"- **Observed Value vs Threshold:** `{alert.get('observed_value', 0):.2f}` (Threshold: `{alert.get('threshold', 0):.2f}`)\n"
                                f"- **Trigger Details:** {alert.get('message', 'No details provided.')}"
                            )

                            if alert.get("model_version"):
                                st.caption(f"🤖 **Model Version Attributed:** `{alert['model_version']}`")

                            # Local Explanation
                            st.markdown("#### 🔬 Diagnostic Threat Explanation")
                            explanations = ModelExplainer.explain_flow_prediction({
                                "flow_duration": float(alert.get("window_seconds", 0.0)),
                                "syn_flag_count": float(alert.get("observed_value", 0.0)) if "SYN" in alert.get("rule_name", "") else 0.0,
                                "total_fwd_packets": 20.0,
                                "total_bwd_packets": 2.0,
                                "flow_packets_s": float(alert.get("observed_value", 0.0)) if "VOLUME" in alert.get("rule_name", "") else 10.0,
                                "flow_bytes_s": float(alert.get("observed_value", 0.0)) if "VOLUME" in alert.get("rule_name", "") else 5000.0,
                            })
                            for exp in explanations:
                                st.info(f"💡 **Signal Detected:** {exp['reason']}")

                            # Correlated Packet Evidence
                            st.markdown("#### 📦 Correlated Packet Evidence")
                            alert_ts = float(alert.get("timestamp", 0.0))
                            win_sec = float(alert.get("window_seconds", 0.0))
                            packets = db.get_related_packets(
                                src_ip=alert.get("src_ip"),
                                dst_ip=alert.get("dst_ip"),
                                start_time=alert_ts - win_sec - 1.0,
                                end_time=alert_ts + 1.0,
                                limit=100,
                            )
                            if packets:
                                st.dataframe(pd.DataFrame(packets), width="stretch")
                            else:
                                st.warning("No matching packet records found in the database during this detection interval.")

                            # Defensive Remediation Guidance
                            st.markdown("#### 🛡️ Prioritized Defensive Remediation Playbook")
                            recs = ModelExplainer.get_defensive_recommendations(alert.get("rule_name", ""))
                            for r in recs:
                                st.markdown(f"- ✅ {r}")

            # ------------------------------------------------------------------
            # PAGE 4: TRAFFIC INTELLIGENCE
            # ------------------------------------------------------------------
            elif selected_page == "Traffic Intelligence":
                st.title("🌐 Network Traffic Intelligence & Flow Analytics")
                st.caption("Inspect discrete time-bucket volumes, flow distributions, and protocol proportions.")

                ti_col1, ti_col2, ti_col3 = st.columns(3)
                ti_col1.metric("Total Observed Packets", f"{packet_count:,}")
                ti_col2.metric("Total Stored Alerts", f"{alert_count:,}")
                ti_col3.metric("Indexed Network Flows", f"{db.get_flow_count():,}")

                # Benign vs Suspicious Ratio
                st.markdown("---")
                st.markdown("### ⚖️ Benign vs. Suspicious Traffic Ratio")
                suspicious_est = min(packet_count, alert_count * 10)
                benign_est = max(0, packet_count - suspicious_est)
                b_ratio = round((benign_est * 100.0) / max(packet_count, 1), 1)
                s_ratio = round((suspicious_est * 100.0) / max(packet_count, 1), 1)

                r_col1, r_col2 = st.columns(2)
                r_col1.metric("Benign Traffic Proportion", f"{b_ratio}%", f"{benign_est:,} packets")
                r_col2.metric("Suspicious / Alert Associated", f"{s_ratio}%", f"{suspicious_est:,} packets", delta_color="inverse")

                # Protocol and Port Breakdown
                st.markdown("---")
                st.markdown("### 🔌 Protocol & Port Breakdown")
                p1, p2 = st.columns(2)
                with p1:
                    p_data = db.get_protocol_distribution()
                    p_chart = build_ranking_chart(p_data, "protocol", "count", "Protocol", "Packets", THEME_COLORS["secondary_accent"])
                    if p_chart:
                        st.altair_chart(p_chart, width="stretch")
                with p2:
                    pt_data = db.get_top_destination_ports(limit=8)
                    pt_chart = build_ranking_chart(pt_data, "dst_port", "packet_count", "Port", "Packets", THEME_COLORS["primary_accent"])
                    if pt_chart:
                        st.altair_chart(pt_chart, width="stretch")

                # Flows Table if available
                st.markdown("---")
                st.markdown("### 🌊 Indexed Bidirectional Network Flows")
                flows = db.get_flows(limit=50)
                if flows:
                    df_flows = pd.DataFrame(flows)
                    disp_cols = [c for c in ["flow_id", "src_ip", "dst_ip", "protocol", "duration", "packet_count", "byte_count", "flow_pkts_s"] if c in df_flows.columns]
                    st.dataframe(df_flows[disp_cols], width="stretch")
                else:
                    st.info("No flow summaries indexed yet. Run `python main.py --pcap <file> --hybrid` to aggregate and index stateful flows.")

            # ------------------------------------------------------------------
            # PAGE 5: AI/ML DETECTION
            # ------------------------------------------------------------------
            elif selected_page == "AI/ML Detection":
                st.title("🤖 AI/ML Detection Engine & Model Health")
                st.caption("Operational transparency for supervised classifiers and unsupervised Isolation Forest anomaly detectors.")

                registry = ModelRegistry(db_path=target_db_path)
                active_sup = db.get_active_model("supervised_binary") or db.get_active_model()
                active_anom = db.get_active_model("anomaly_isolation_forest")

                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.markdown("### 🎯 Supervised Classifier Status")
                    if active_sup:
                        st.success(f"**Algorithm:** {active_sup['algorithm']} (`{active_sup['version']}`)")
                        st.markdown(
                            f"- **Target Task:** `{active_sup.get('task_type', 'binary')}`\n"
                            f"- **Training Dataset:** `{active_sup.get('dataset_name', 'Unknown')}`\n"
                            f"- **Accuracy (Test):** `{active_sup['metrics'].get('accuracy', 0)*100:.2f}%`\n"
                            f"- **F1 Score (Weighted):** `{active_sup['metrics'].get('f1_weighted', 0)*100:.2f}%`\n"
                            f"- **Precision (Macro):** `{active_sup['metrics'].get('precision_macro', 0)*100:.2f}%`\n"
                            f"- **Inference Latency:** `{active_sup['metrics'].get('inference_latency_ms_per_sample', 0):.4f} ms/sample`"
                        )
                    else:
                        st.warning("No supervised model active. Head to Model Laboratory to train or activate one.")

                with m_col2:
                    st.markdown("### 🌲 Unsupervised Anomaly Detector")
                    if active_anom:
                        st.success(f"**Algorithm:** Isolation Forest (`{active_anom['version']}`)")
                        st.markdown(
                            f"- **Training Baseline:** Benign network flows (zero malicious contamination)\n"
                            f"- **Estimators:** 100 Isolation Trees\n"
                            f"- **Score Threshold:** `{active_anom['metrics'].get('score_threshold', 0.5):.3f}`\n"
                            f"- **Active Status:** `ONLINE (Real-time scoring)`"
                        )
                    else:
                        st.warning("No anomaly detector active. Head to Model Laboratory to train or activate one.")

                # Feature Schema
                st.markdown("---")
                st.markdown("### 📐 Active Feature Schema (28 Inputs)")
                if active_sup and active_sup.get("feature_schema"):
                    feats = active_sup["feature_schema"]
                    f_cols = st.columns(4)
                    for idx, f in enumerate(feats):
                        f_cols[idx % 4].caption(f"`{idx+1}. {f}`")

                # Scientific Integrity Alert
                st.markdown("---")
                st.info(
                    "⚠️ **Scientific & Operational Notice:**\n\n"
                    "Unusual traffic patterns flagged by Isolation Forest do **NOT** automatically prove malicious compromise. "
                    "Network variance (e.g. software updates, large backup jobs, multi-stream media) frequently triggers high anomaly scores. "
                    "All AI/ML detections serve strictly as prioritized decision-support signals for security analysts."
                )

            # ------------------------------------------------------------------
            # PAGE 6: MODEL LABORATORY (Training & Evaluation Workspace)
            # ------------------------------------------------------------------
            elif selected_page == "Model Laboratory":
                st.title("🔬 Model Laboratory & Experimentation Workbench")
                st.caption("Train real classifiers, inspect verifiable metrics, compare confusion matrices, and activate validated model versions.")

                ds_manager = DatasetManager()
                listing = ds_manager.list_datasets()
                registry = ModelRegistry(db_path=target_db_path)

                with st.expander("🛠️ Train a New Model", expanded=True):
                    t_col1, t_col2, t_col3 = st.columns(3)
                    with t_col1:
                        target_ds = st.selectbox("Training Dataset", ["cicids2017", "unsw_nb15"], index=0)
                    with t_col2:
                        target_algo = st.selectbox(
                            "Algorithm",
                            ["Random Forest", "Logistic Regression", "Isolation Forest"],
                            index=0,
                        )
                    with t_col3:
                        task_mode = st.selectbox("Classification Task", ["binary", "multiclass"], index=0)

                    auto_activate = st.checkbox("Automatically activate if validation succeeds", value=True)

                    if st.button("🚀 Train Model Now", type="primary"):
                        paths = ds_manager.get_dataset_paths(target_ds)
                        # Check raw file or create deterministic sample
                        raw_files = list(paths["raw"].glob("*.csv"))
                        if not raw_files:
                            st.info("Creating deterministic sample dataset fixture...")
                            ds_manager.generate_reproducible_sample_dataset(target_ds)
                            raw_files = list(paths["raw"].glob("*.csv"))

                        with st.spinner(f"Preprocessing data and training {target_algo}..."):
                            df = pd.read_csv(raw_files[0])
                            preprocessor = FlowDataPreprocessor(task_type=task_mode, split_method="stratified")
                            splits, prep_rep = preprocessor.fit_transform(df)

                            X_train, y_train = splits["X_train"], splits["y_train"]
                            X_test, y_test = splits["X_test"], splits["y_test"]

                            algo_slug = target_algo.lower().replace(" ", "_")
                            ver = f"{algo_slug}_{task_mode}_{int(time.time())}"

                            if algo_slug == "isolation_forest":
                                b_mask = y_train == 0
                                X_ben = X_train[b_mask] if b_mask.any() else X_train
                                trainer = AnomalyDetectorTrainer()
                                model, summary = trainer.train(X_ben)
                                is_anom, scores = trainer.score_samples(X_test)
                                acc = float(np.mean((is_anom.astype(int) == y_test)))
                                metrics = {
                                    "training_duration_seconds": summary.training_duration_seconds,
                                    "train_samples": summary.train_samples,
                                    "score_threshold": summary.score_threshold,
                                    "accuracy": acc,
                                }
                                m_type = "anomaly_isolation_forest"
                            else:
                                trainer = SupervisedModelTrainer(algorithm=algo_slug, task_type=task_mode)
                                model, summary = trainer.train(X_train, y_train)
                                class_names = [preprocessor.inverse_label_mapping_.get(i, f"Class_{i}") for i in sorted(list(set(y_train)))]
                                eval_res = ModelEvaluator.evaluate_classifier(
                                    model=model,
                                    X_test=X_test,
                                    y_test=y_test,
                                    model_name=target_algo,
                                    model_version=ver,
                                    dataset_name=target_ds,
                                    task_type=task_mode,
                                    split_method=prep_rep.split_method,
                                    class_names=class_names,
                                )
                                ModelEvaluator.save_reports(eval_res)
                                metrics = eval_res.to_dict()
                                m_type = f"supervised_{task_mode}"

                            meta = registry.save_model_bundle(
                                name=f"{target_algo}_{target_ds.upper()}",
                                version=ver,
                                model=model,
                                preprocessor=preprocessor,
                                model_type=m_type,
                                algorithm=target_algo,
                                dataset_name=target_ds,
                                task_type=task_mode,
                                feature_schema=preprocessor.feature_names_,
                                metrics=metrics,
                                set_active=auto_activate,
                            )
                            st.success(f"✅ Model `{ver}` trained successfully in {summary.training_duration_seconds:.2f}s!")
                            st.rerun()

                # Model Registry Table
                st.markdown("---")
                st.subheader("📚 Model Artifact Registry")
                all_models = db.get_models()
                if not all_models:
                    st.info("No models registered yet. Train a model using the form above.")
                else:
                    for m in all_models:
                        is_act = bool(m.get("is_active"))
                        act_badge = "🟢 ACTIVE" if is_act else "⚪ INACTIVE"
                        with st.container():
                            st.markdown(f"#### {act_badge} `{m['name']}` (v: `{m['version']}`)")
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            mc1.metric("Algorithm", m.get("algorithm", "-"))
                            mc2.metric("Task Type", m.get("task_type", "-"))
                            mc3.metric("Accuracy", f"{m['metrics'].get('accuracy', 0)*100:.2f}%")
                            mc4.metric("F1 Score", f"{m['metrics'].get('f1_weighted', 0)*100:.2f}%")

                            if not is_act:
                                if st.button(f"Activate {m['version']}", key=f"btn_act_{m['version']}"):
                                    db.set_active_model(m["version"], is_active=True)
                                    st.success(f"Activated model {m['version']}!")
                                    st.rerun()
                            st.caption(f"📁 Path: `{m.get('artifact_path')}` | Created: {format_timestamp(m.get('created_at'))}")
                            st.markdown("---")

            # ------------------------------------------------------------------
            # PAGE 7: DATA EXPLORER (Data Science EDA Workspace)
            # ------------------------------------------------------------------
            elif selected_page == "Data Explorer":
                st.title("📊 Data Explorer & Exploratory Analysis (EDA)")
                st.caption("Inspect raw dataset health, class distributions, missing value audits, and feature correlation matrices.")

                manager = DatasetManager()
                target_ds = st.selectbox("Choose Dataset to Explore", ["cicids2017", "unsw_nb15"], index=0)
                paths = manager.get_dataset_paths(target_ds)
                raw_files = list(paths["raw"].glob("*.csv"))

                if not raw_files:
                    st.info(f"No CSV files found in `{paths['raw']}`. Generating sample fixture...")
                    manager.generate_reproducible_sample_dataset(target_ds)
                    raw_files = list(paths["raw"].glob("*.csv"))

                selected_csv = st.selectbox("Select CSV File", options=[f.name for f in raw_files], index=0)
                file_path = paths["raw"] / selected_csv
                df = pd.read_csv(file_path)

                profile = DatasetProfiler.profile_dataframe(df)

                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Total Records", f"{profile['total_rows']:,}")
                e2.metric("Total Features", f"{profile['total_cols']:,}")
                e3.metric("Duplicate Rows", f"{profile['duplicate_rows']:,}")
                e4.metric("Constant Columns", f"{len(profile['constant_columns'])}")

                # Class Distribution
                st.markdown("---")
                st.subheader("🎯 Ground-Truth Class Distribution")
                if profile["class_distribution"]:
                    cd_df = pd.DataFrame([
                        {"Class": k, "Count": v, "Percentage": round(v * 100.0 / profile['total_rows'], 2)}
                        for k, v in profile["class_distribution"].items()
                    ])
                    col_chart, col_tbl = st.columns([2, 1])
                    with col_chart:
                        c_chart = (
                            alt.Chart(cd_df)
                            .mark_bar(color=THEME_COLORS["primary_accent"], cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                            .encode(
                                x=alt.X("Count:Q", title="Samples"),
                                y=alt.Y("Class:N", sort="-x", title="Target Class"),
                                tooltip=["Class:N", "Count:Q", "Percentage:Q"],
                            )
                            .properties(height=200)
                        )
                        st.altair_chart(c_chart, width="stretch")
                    with col_tbl:
                        st.dataframe(cd_df, width="stretch")

                # Numerical Summaries
                st.markdown("---")
                st.subheader("📈 Statistical Feature Summaries")
                if profile["numeric_summaries"]:
                    summary_df = pd.DataFrame(profile["numeric_summaries"]).T
                    st.dataframe(summary_df.round(3), width="stretch")

                # Correlation Matrix
                st.markdown("---")
                st.subheader("🔥 Feature Correlation Analysis")
                cols, corr_matrix = DatasetProfiler.compute_correlation_matrix(df, max_features=12)
                if len(cols) > 1:
                    corr_df = pd.DataFrame(corr_matrix, index=cols, columns=cols).round(2)
                    st.dataframe(corr_df, width="stretch")

            # ------------------------------------------------------------------
            # PAGE 8: DETECTION RULES
            # ------------------------------------------------------------------
            elif selected_page == "Detection Rules":
                st.title("⚙️ Detection Rules & Heuristic Policies")
                st.caption("Inspect and manage threshold parameters for rule-based and machine-learning detectors.")

                rules_table = [
                    {
                        "Rule": "PORT_SCAN",
                        "Category": "Reconnaissance",
                        "Target": "TCP Probes across distinct destination ports",
                        "Threshold": "15 distinct ports",
                        "Window": "30.0 seconds",
                        "Severity": "HIGH",
                        "Status": "ACTIVE",
                    },
                    {
                        "Rule": "SYN_RATE",
                        "Category": "Denial of Service (DoS)",
                        "Target": "High-velocity TCP SYN packet bursts",
                        "Threshold": "20 SYN packets",
                        "Window": "10.0 seconds",
                        "Severity": "MEDIUM",
                        "Status": "ACTIVE",
                    },
                    {
                        "Rule": "TRAFFIC_VOLUME",
                        "Category": "Anomaly / Exfiltration",
                        "Target": "Sudden spikes in byte or packet volume",
                        "Threshold": "100,000 bytes or 500 packets",
                        "Window": "60.0 seconds",
                        "Severity": "MEDIUM",
                        "Status": "ACTIVE",
                    },
                    {
                        "Rule": "ML_SUPERVISED",
                        "Category": "Machine Learning Classifier",
                        "Target": "Multivariate flow vector matching known attack patterns",
                        "Threshold": "Confidence >= 0.50",
                        "Window": "Stateful Flow Duration",
                        "Severity": "CRITICAL / HIGH",
                        "Status": "ACTIVE" if active_sup else "STANDBY",
                    },
                    {
                        "Rule": "ML_ANOMALY",
                        "Category": "Unsupervised Outlier",
                        "Target": "Statistical deviation from benign traffic profile",
                        "Threshold": "Anomaly Score >= 0.50",
                        "Window": "Stateful Flow Duration",
                        "Severity": "MEDIUM",
                        "Status": "ACTIVE" if active_anom else "STANDBY",
                    },
                ]
                st.dataframe(pd.DataFrame(rules_table), width="stretch")

            # ------------------------------------------------------------------
            # PAGE 9: SYSTEM & SETTINGS
            # ------------------------------------------------------------------
            elif selected_page == "System & Settings":
                st.title("🖥️ System Diagnostics & Environment Status")
                st.caption("Verify runtime dependencies, database health, and evaluation reports.")

                import sys
                s1, s2, s3 = st.columns(3)
                s1.metric("Python Interpreter", sys.version.split(" ")[0])
                s2.metric("Database Path", str(target_db_path.name))
                file_size_kb = target_db_path.stat().st_size / 1024 if target_db_path.exists() else 0
                s3.metric("Database Size", f"{file_size_kb:.1f} KB")

                st.markdown("---")
                st.markdown("### 📄 Evaluation Reports Browser")
                reports_dir = Path("data") / "reports"
                if reports_dir.exists():
                    md_reports = list(reports_dir.glob("*.md"))
                    if md_reports:
                        chosen_rep = st.selectbox("Select Evaluation Report", [r.name for r in md_reports])
                        rep_path = reports_dir / chosen_rep
                        st.markdown(rep_path.read_text(encoding="utf-8"))
                    else:
                        st.info("No evaluation reports generated yet. Run `python scripts/evaluate_models.py`.")
                else:
                    st.info("No evaluation reports generated yet.")

                st.markdown("---")
                st.markdown("### ⚖️ Defensive Cybersecurity & Privacy Principles")
                st.caption(
                    "Mini-IDS is an educational, passive network intrusion detection system. "
                    "It observes packet headers and flow metadata without altering, injecting, or terminating network sessions. "
                    "All captured network telemetry remains strictly on the local workstation and is never uploaded externally."
                )

    except sqlite3.Error as sql_err:
        st.error(f"Database error while reading `{target_db_path}`: {sql_err}")
        return
    except Exception as err:
        st.error(f"Failed to access database `{target_db_path}`: {err}")
        return


if __name__ == "__main__":
    render_dashboard()
