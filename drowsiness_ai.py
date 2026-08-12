import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sys
import json
from datetime import datetime


DB_NAME = "drowsiness_telemetry.db"
OUTPUT_FILE = "driver_telemetry_dashboard.html"

THEME = {
    "bg_deep":      "#080c14",
    "bg_card":      "#0d1421",
    "bg_border":    "#1a2540",
    "grid":         "#111827",
    "primary":      "#38bdf8",   
    "danger":       "#f43f5e",   
    "warning":      "#fbbf24",   
    "success":      "#34d399",   
    "accent":       "#a78bfa",  
    "muted":        "#475569",   
    "text_bright":  "#f1f5f9",
    "text_sub":     "#94a3b8",
}

def load_telemetry_data():
    """Extracts full-day data and the most recent session."""
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM telemetry ORDER BY timestamp ASC", conn)
        conn.close()

        if df.empty:
            raise ValueError("Database is empty.")

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        today = pd.Timestamp.now().normalize()
        daily_df = df[df["timestamp"] >= today].copy()

        latest_session_id = df.iloc[-1]["session_id"]
        session_df = df[df["session_id"] == latest_session_id].copy()
        session_df["time_sec"] = (
            session_df["timestamp"] - session_df["timestamp"].min()
        ).dt.total_seconds()

        print(f"✓ Daily data loaded: {len(daily_df)} frames.")
        print(f"✓ Active session ({latest_session_id}): {len(session_df)} frames.")
        return daily_df, session_df

    except Exception as e:
        print(f"❌ DB Error: {e}")
        sys.exit(1)


def engineer_features(session_df: pd.DataFrame) -> pd.DataFrame:
    """Derives richer signals from raw telemetry."""
    df = session_df.copy()

    for col, w in [("ear", 30), ("mar", 20), ("fatigue_score", 60),
                   ("blink_rate", 45), ("pitch", 20), ("yaw", 20)]:
        if col in df.columns:
            df[f"{col}_smooth"] = df[col].rolling(window=w, min_periods=1).mean()

    df["fatigue_delta"]  = df["fatigue_score"].diff().fillna(0)
    df["ear_velocity"]   = df["ear"].diff().fillna(0)

   
    ear_norm     = df["ear"].clip(0.1, 0.45).pipe(lambda s: (s - 0.1) / 0.35)
    fatigue_norm = 1 - (df["fatigue_score"].clip(0, 1))
    blink_norm   = df["blink_rate"].clip(5, 30).pipe(lambda s: 1 - (s - 5) / 25)
    df["alertness_index"] = (0.45 * ear_norm + 0.40 * fatigue_norm + 0.15 * blink_norm).clip(0, 1)

    return df



def evaluate_tracking_accuracy(df: pd.DataFrame):
    """Isolation Forest on scaled biometric features to detect tracking glitches."""
    print("🧠 Running AI Anomaly Detection …")

    features = [c for c in ["ear", "mar", "pitch", "yaw"] if c in df.columns]
    X_raw = df[features].bfill().ffill()

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = IsolationForest(
        n_estimators=150, contamination=0.025,
        max_samples="auto", random_state=42
    )
    df["anomaly_score"] = model.fit_predict(X)        # -1 = outlier
    df["anomaly_conf"]  = -model.score_samples(X)     # higher = more anomalous
    df["is_glitch"]     = (df["anomaly_score"] == -1).astype(int)

    accuracy_pct = (1 - df["is_glitch"].mean()) * 100
    print(f"🎯 Tracking stability: {accuracy_pct:.2f}%")
    return df, accuracy_pct



def compute_kpis(daily_df: pd.DataFrame, session_df: pd.DataFrame, accuracy_pct: float) -> dict:
    dur_min = session_df["time_sec"].max() / 60 if not session_df.empty else 0
    avg_fatigue   = session_df["fatigue_score"].mean() if "fatigue_score" in session_df else 0
    peak_fatigue  = session_df["fatigue_score"].max()  if "fatigue_score" in session_df else 0
    microsleeps   = int(session_df["microsleep_events"].iloc[-1]) if "microsleep_events" in session_df else 0
    avg_ear       = session_df["ear"].mean() if "ear" in session_df else 0
    alert_idx     = session_df["alertness_index"].mean() if "alertness_index" in session_df else 0
    glitch_count  = int(session_df["is_glitch"].sum())

    risk_score = min(100, avg_fatigue * 60 + microsleeps * 8 + max(0, 0.25 - avg_ear) * 120)

    return {
        "duration_min":   f"{dur_min:.1f}",
        "avg_fatigue":    f"{avg_fatigue:.2f}",
        "peak_fatigue":   f"{peak_fatigue:.2f}",
        "microsleeps":    str(microsleeps),
        "avg_ear":        f"{avg_ear:.3f}",
        "alertness":      f"{alert_idx * 100:.1f}%",
        "tracking_acc":   f"{accuracy_pct:.1f}%",
        "glitch_count":   str(glitch_count),
        "risk_score":     f"{risk_score:.0f}/100",
        "risk_color":     THEME["danger"] if risk_score > 60 else THEME["warning"] if risk_score > 30 else THEME["success"],
    }



def generate_dashboard(daily_df: pd.DataFrame, session_df: pd.DataFrame, accuracy_pct: float):
    print("🎨 Rendering modern dashboard …")

    T = THEME
    kpis = compute_kpis(daily_df, session_df, accuracy_pct)

   
    daily_df = daily_df.copy()
    daily_df["fatigue_trend"] = daily_df["fatigue_score"].rolling(150, min_periods=1).mean()
    daily_df["fatigue_upper"] = daily_df["fatigue_score"].rolling(150, min_periods=1).quantile(0.85)
    daily_df["fatigue_lower"] = daily_df["fatigue_score"].rolling(150, min_periods=1).quantile(0.15)

   
    fig = make_subplots(
        rows=5, cols=2,
        specs=[
            [{"colspan": 2, "secondary_y": False}, None],
            [{"colspan": 2, "secondary_y": True},  None],
            [{"colspan": 2, "type": "bar"},        None], 
            [{"colspan": 2, "secondary_y": True},  None],
            [{"colspan": 2, "secondary_y": False}, None],
        ],
        subplot_titles=[
            "FULL-DAY FATIGUE PROGRESSION",
            "SESSION · FATIGUE SCORE  vs  EYE ASPECT RATIO",
            "CUMULATIVE MICROSLEEP EVENTS", # Pie chart title removed
            "BLINK RATE  vs  HEAD PITCH DYNAMICS",
            f"AI TRACKING QUALITY — {accuracy_pct:.2f}% STABLE",
        ],
        vertical_spacing=0.072,
        horizontal_spacing=0.06,
    )

    
    fig.add_trace(go.Scatter(
        x=pd.concat([daily_df["timestamp"], daily_df["timestamp"].iloc[::-1]]),
        y=pd.concat([daily_df["fatigue_upper"], daily_df["fatigue_lower"].iloc[::-1]]),
        fill="toself", fillcolor=f"rgba(167,139,250,0.08)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=daily_df["timestamp"], y=daily_df["fatigue_trend"],
        mode="lines", name="Daily Fatigue Trend",
        line=dict(color=T["accent"], width=2.5),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.12)",
        hovertemplate="<b>%{x|%H:%M:%S}</b><br>Fatigue: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    
    fig.add_trace(go.Scatter(
        x=session_df["time_sec"], y=session_df["fatigue_score_smooth"],
        mode="lines", name="Fatigue (smooth)",
        line=dict(color=T["danger"], width=2.5),
        fill="tozeroy", fillcolor="rgba(244,63,94,0.08)",
        hovertemplate="<b>%{x:.1f}s</b><br>Fatigue: %{y:.3f}<extra></extra>",
    ), row=2, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=session_df["time_sec"], y=session_df["ear_smooth"],
        mode="lines", name="EAR (smooth)",
        line=dict(color=T["success"], width=2, dash="dot"),
        hovertemplate="<b>%{x:.1f}s</b><br>EAR: %{y:.4f}<extra></extra>",
    ), row=2, col=1, secondary_y=True)

    
    fig.add_trace(go.Scatter(
        x=session_df["time_sec"], y=session_df["alertness_index"],
        mode="lines", name="Alertness Index",
        line=dict(color=T["primary"], width=1.2, dash="dashdot"),
        opacity=0.6,
        hovertemplate="<b>%{x:.1f}s</b><br>Alertness: %{y:.2f}<extra></extra>",
    ), row=2, col=1, secondary_y=True)


   
    fig.add_trace(go.Bar(
        x=session_df["time_sec"],
        y=session_df["microsleep_events"],
        name="Cumulative Microsleeps",
        marker=dict(
            color=session_df["microsleep_events"],
            colorscale=[[0, T["success"]], [0.5, T["warning"]], [1, T["danger"]]],
            showscale=False,
        ),
        marker_line_width=0, opacity=0.9,
        hovertemplate="<b>%{x:.1f}s</b><br>Microsleeps: %{y}<extra></extra>",
    ), row=3, col=1)

  
    fig.add_trace(go.Scatter(
        x=session_df["time_sec"], y=session_df["blink_rate_smooth"],
        mode="lines", name="Blink Rate /min",
        line=dict(color=T["primary"], width=2),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.07)",
        hovertemplate="<b>%{x:.1f}s</b><br>Blink Rate: %{y:.1f}/min<extra></extra>",
    ), row=4, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=session_df["time_sec"], y=session_df["pitch_smooth"],
        mode="lines", name="Head Pitch °",
        line=dict(color=T["warning"], width=1.8, dash="dash"),
        hovertemplate="<b>%{x:.1f}s</b><br>Pitch: %{y:.1f}°<extra></extra>",
    ), row=4, col=1, secondary_y=True)

   
    fig.add_trace(go.Scatter(
        x=session_df["time_sec"], y=session_df["ear"],
        mode="lines", name="Raw EAR",
        line=dict(color=f"rgba(148,163,184,0.25)", width=1),
        hoverinfo="skip",
    ), row=5, col=1)

    
    fig.add_trace(go.Scatter(
        x=session_df["time_sec"], y=session_df["ear_smooth"],
        mode="lines", name="EAR Baseline",
        line=dict(color=f"rgba(56,189,248,0.55)", width=1.5),
        hoverinfo="skip",
    ), row=5, col=1)

    glitches = session_df[session_df["is_glitch"] == 1]
    if not glitches.empty:
        fig.add_trace(go.Scatter(
            x=glitches["time_sec"], y=glitches["ear"],
            mode="markers", name="Tracking Glitch",
            marker=dict(
                color=glitches["anomaly_conf"],
                colorscale=[[0, T["warning"]], [1, T["danger"]]],
                size=10 + glitches["anomaly_conf"].clip(0, 2) * 5,
                symbol="x-open",
                line=dict(color="white", width=1.2),
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Anomaly<br>Confidence",
                        font=dict(color=T["text_sub"], size=10)
                    ),
                    tickfont=dict(color=T["text_sub"], size=10),
                    len=0.18, y=0.02, x=1.0,
                ),
            ),
            hovertemplate="<b>Glitch @ %{x:.1f}s</b><br>EAR: %{y:.4f}<br>Conf: %{marker.color:.3f}<extra></extra>",
        ), row=5, col=1)

   
    now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    fig.update_layout(
        title=dict(
            text=(
                f"<b style='letter-spacing:3px'>DROWSINESS TELEMETRY</b>"
                f"<br><span style='font-size:12px;color:{T['text_sub']};letter-spacing:2px'>"
                f"REPORT · {now_str}</span>"
            ),
            x=0.03, y=0.985,
            font=dict(size=22, color=T["text_bright"], family="'Courier New', monospace"),
        ),
        template="plotly_dark",
        height=1950,
        margin=dict(l=60, r=60, t=130, b=60),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=T["bg_card"],
            font=dict(color=T["text_bright"], size=12),
            bordercolor=T["bg_border"],
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.015,
            xanchor="right",  x=0.98,
            bgcolor="rgba(13,20,33,0.7)",
            bordercolor=T["bg_border"],
            borderwidth=1,
            font=dict(color=T["text_sub"], size=11),
        ),
        plot_bgcolor=T["bg_deep"],
        paper_bgcolor=T["bg_deep"],
        font=dict(family="'Courier New', Consolas, monospace", color=T["text_sub"], size=11),
    )

   
    grid_cfg = dict(showgrid=True, gridwidth=1, gridcolor=T["grid"], zeroline=False)
    fig.update_xaxes(**grid_cfg, tickfont=dict(size=10))
    fig.update_yaxes(**grid_cfg, tickfont=dict(size=10))

    fig.update_xaxes(title_text="Time of Day",            row=1, col=1, title_font_size=11)
    fig.update_xaxes(title_text="Session Time (seconds)", row=5, col=1, title_font_size=11)

    fig.update_yaxes(title_text="Fatigue Score",  row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="EAR / Alertness",row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Blink Rate /min",row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Pitch (°)",      row=4, col=1, secondary_y=True)

  
    for ann in fig.layout.annotations:
        ann.font = dict(
            size=10,
            color=T["text_sub"],
            family="'Courier New', monospace",
        )
        ann.update(x=ann.x)

   
    plot_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config=dict(displayModeBar=True, responsive=True),
    )

    kpi_items = [
        ("⏱ SESSION",         kpis["duration_min"] + " min",      T["primary"]),
        ("💤 MICROSLEEPS",    kpis["microsleeps"],                  T["warning"]),
        ("😴 AVG FATIGUE",    kpis["avg_fatigue"],                  T["accent"]),
        ("🔺 PEAK FATIGUE",   kpis["peak_fatigue"],                 T["danger"]),
        ("👁 AVG EAR",        kpis["avg_ear"],                      T["success"]),
        ("⚡ ALERTNESS",      kpis["alertness"],                    T["primary"]),
        ("🤖 TRACKING",       kpis["tracking_acc"],                 T["success"]),
        ("🚨 RISK SCORE",     kpis["risk_score"],                   kpis["risk_color"]),
    ]

    kpi_cards_html = "\n".join(
        f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color}">{value}</div>
            </div>"""
        for label, value, color in kpi_items
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Drowsiness Telemetry Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:         {T["bg_deep"]};
      --bg-card:    {T["bg_card"]};
      --border:     {T["bg_border"]};
      --primary:    {T["primary"]};
      --danger:     {T["danger"]};
      --warning:    {T["warning"]};
      --success:    {T["success"]};
      --accent:     {T["accent"]};
      --text:       {T["text_bright"]};
      --text-sub:   {T["text_sub"]};
      --mono:       'Space Mono', 'Courier New', monospace;
      --sans:       'DM Sans', sans-serif;
    }}

    html, body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      min-height: 100vh;
      overflow-x: hidden;
    }}

    /* ── SCANLINE OVERLAY ── */
    body::before {{
      content: '';
      position: fixed; inset: 0; z-index: 0;
      background: repeating-linear-gradient(
        0deg,
        rgba(0,0,0,0.03) 0px,
        rgba(0,0,0,0.03) 1px,
        transparent 1px,
        transparent 3px
      );
      pointer-events: none;
    }}

    .wrapper {{
      position: relative; z-index: 1;
      max-width: 1600px;
      margin: 0 auto;
      padding: 40px 32px 80px;
    }}

    /* ── HEADER ── */
    .header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      border-bottom: 1px solid var(--border);
      padding-bottom: 24px;
      margin-bottom: 36px;
    }}
    .header-left h1 {{
      font-family: var(--mono);
      font-size: clamp(20px, 3vw, 30px);
      font-weight: 700;
      letter-spacing: 4px;
      color: var(--text);
      line-height: 1;
    }}
    .header-left h1 span {{ color: var(--primary); }}
    .header-left .subtitle {{
      font-family: var(--mono);
      font-size: 11px;
      color: var(--text-sub);
      letter-spacing: 2px;
      margin-top: 8px;
    }}
    .header-right {{
      text-align: right;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--text-sub);
      letter-spacing: 1px;
      line-height: 1.8;
    }}

    /* ── STATUS BADGE ── */
    .status-badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(52,211,153,0.12);
      border: 1px solid rgba(52,211,153,0.3);
      border-radius: 4px;
      padding: 4px 12px;
      font-family: var(--mono);
      font-size: 10px;
      color: var(--success);
      letter-spacing: 2px;
      margin-top: 8px;
    }}
    .status-badge::before {{
      content: '';
      width: 6px; height: 6px;
      border-radius: 50%;
      background: var(--success);
      animation: pulse 2s ease-in-out infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; transform: scale(1); }}
      50%       {{ opacity: 0.4; transform: scale(0.8); }}
    }}

    /* ── KPI GRID ── */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 16px;
      margin-bottom: 40px;
    }}
    .kpi-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 20px 18px 16px;
      position: relative;
      overflow: hidden;
      transition: border-color 0.2s, transform 0.2s;
    }}
    .kpi-card::after {{
      content: '';
      position: absolute;
      bottom: 0; left: 0;
      width: 100%; height: 2px;
      background: linear-gradient(90deg, transparent, var(--primary), transparent);
      opacity: 0;
      transition: opacity 0.3s;
    }}
    .kpi-card:hover {{
      border-color: rgba(56,189,248,0.4);
      transform: translateY(-2px);
    }}
    .kpi-card:hover::after {{ opacity: 1; }}
    .kpi-label {{
      font-family: var(--mono);
      font-size: 9.5px;
      letter-spacing: 2px;
      color: var(--text-sub);
      margin-bottom: 10px;
    }}
    .kpi-value {{
      font-family: var(--mono);
      font-size: clamp(18px, 2.5vw, 26px);
      font-weight: 700;
      line-height: 1;
    }}

    /* ── CHART SECTION ── */
    .chart-wrapper {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      margin-bottom: 12px;
    }}
    .chart-wrapper > div {{
      border-radius: 6px;
      overflow: hidden;
    }}

    /* ── FOOTER ── */
    .footer {{
      margin-top: 60px;
      padding-top: 24px;
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--mono);
      font-size: 10px;
      color: var(--text-sub);
      letter-spacing: 1.5px;
    }}

    /* ── SECTION LABELS ── */
    .section-label {{
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 3px;
      color: var(--text-sub);
      text-transform: uppercase;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .section-label::after {{
      content: '';
      flex: 1;
      height: 1px;
      background: var(--border);
    }}
  </style>
</head>
<body>
<div class="wrapper">

  <header class="header">
    <div class="header-left">
      <h1>DROWSINESS <span>TELEMETRY</span></h1>
      <div class="subtitle">DRIVER SAFETY ANALYTICS SYSTEM</div>
      <div class="status-badge">SYSTEM OPERATIONAL</div>
    </div>
    <div class="header-right">
      REPORT GENERATED<br>
      <strong style="color:var(--text)">{now_str}</strong><br>
      <span style="color:var(--primary)">TRACKING · {kpis["tracking_acc"]} STABLE</span>
    </div>
  </header>

  <div class="section-label">KEY PERFORMANCE INDICATORS</div>
  <div class="kpi-grid">
    {kpi_cards_html}
  </div>

  <div class="section-label">TELEMETRY ANALYTICS</div>
  <div class="chart-wrapper">
    {plot_html}
  </div>

  <footer class="footer">
    <span>DROWSINESS DETECTION SYSTEM · AI-ENHANCED TELEMETRY</span>
    <span>FRAMES ANALYSED · SESSION</span>
  </footer>

</div>
</body>
</html>"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"✨ Dashboard saved → {OUTPUT_FILE}")
    print("   Open in your browser to explore the interactive analytics.")



if __name__ == "__main__":
    daily_data, session_data = load_telemetry_data()
    session_data = engineer_features(session_data)
    session_data, accuracy = evaluate_tracking_accuracy(session_data)
    generate_dashboard(daily_data, session_data, accuracy)