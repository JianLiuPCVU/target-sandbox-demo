import html
import math
import re
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, SeasonalNaive
except ImportError:
    StatsForecast = None
    AutoETS = None
    SeasonalNaive = None

st.set_page_config(page_title="Target Sandbox | Interactive Demo", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {display:none !important;}
.block-container {padding-top: 2.1rem; max-width: 1550px;}
</style>""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).with_name("Sales_Sample.csv")
MODEL_CHOICE = "AutoETS"
PREDICTION_LEVEL = 80

METRICS = {
    "Overall Sales": "overall",
    "Store Sales": "store",
    "Online Sales": "online",
    "Ad Sales": "ad",
}
BASE_METRICS = ["store", "ad", "organic"]
METRIC_LABELS = {
    "store": "Store Sales",
    "ad": "Ad Sales",
    "organic": "Organic Online",
    "online": "Online Sales",
    "overall": "Overall Sales",
}
COL_CANDIDATES = {
    "date": ["date", "day", "business date"],
    "sku": ["sku", "item", "product id"],
    "brand": ["brand"],
    "category": ["category"],
    "store": ["store sales amount", "store sales", "store"],
    "ad": ["ad sales amount", "ad sales", "advertising sales"],
    "organic": ["online organic sales amount", "organic online sales", "online organic sales", "organic sales"],
    "online": ["online sales amount", "online sales"],
    "overall": ["overall sales amount", "overall sales", "total sales"],
}

def default_sandboxes():
    """Each browser session starts with an independent sample Sandbox."""
    return [{
        "id": "sandbox-1",
        "name": "My First Sandbox",
        "brands": [],
        "categories": [],
        "start_date": "2026-10-01",
        "end_date": "2026-12-31",
        "targets": {"overall": 100000000.0},
    }]


def load_sandboxes():
    if "sandboxes_v7" not in st.session_state:
        st.session_state["sandboxes_v7"] = default_sandboxes()
    return st.session_state["sandboxes_v7"]


def save_sandboxes(items):
    # Session-only persistence: never write shared files on the server.
    st.session_state["sandboxes_v7"] = items

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace("_", " ") for c in df.columns]
    found = {}
    for logical, candidates in COL_CANDIDATES.items():
        for col in df.columns:
            if col in candidates:
                found[logical] = col
                break
    return df, found

def parse_money(series):
    # Handles values such as "$ 81,143.22", commas, currency symbols, and blanks.
    cleaned = series.astype(str).str.replace(r"[$,\s]", "", regex=True)
    cleaned = cleaned.replace({"": None, "nan": None, "None": None, "-": None})
    return pd.to_numeric(cleaned, errors="coerce")

def normalize_sku(series):
    s = series.astype(str).str.strip()
    # CSV exports sometimes convert numeric identifiers to strings ending in .0.
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": "", "None": ""})
    return s

def prepare_data(uploaded):
    try:
        raw = pd.read_csv(uploaded, dtype=str)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}")
    raw, found = normalize_columns(raw)
    required = ["date", "sku", "brand", "category", "store", "ad", "organic"]
    missing = [k for k in required if k not in found]
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(
            "CSV is missing required columns: " + missing_names +
            " Required: Date, SKU, Brand, Category, Store Sales Amount, Ad Sales Amount, Online Organic Sales Amount."
        )

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw[found["date"]], errors="coerce")
    df["sku"] = normalize_sku(raw[found["sku"]])
    df["brand"] = raw[found["brand"]].fillna("").astype(str).str.strip()
    df["category"] = raw[found["category"]].fillna("").astype(str).str.strip()
    for key in ["store", "ad", "organic"]:
        df[key] = parse_money(raw[found[key]])
    invalid = df["date"].isna() | df["sku"].eq("")
    for key in ["store", "ad", "organic"]:
        invalid = invalid | df[key].isna()
    invalid_count = int(invalid.sum())
    df = df.loc[~invalid].copy()
    if df.empty:
        raise ValueError("No valid records found. Check the date, SKU, and sales amount columns.")
    df["date"] = df["date"].dt.normalize()
    # Aggregate duplicate rows for the same SKU/day/scope dimensions.
    grouped = df.groupby(["date", "sku", "brand", "category"], as_index=False)[["store", "ad", "organic"]].sum()
    grouped["online"] = grouped["ad"] + grouped["organic"]
    grouped["overall"] = grouped["store"] + grouped["online"]
    return grouped, invalid_count, found

def money(v):
    return f"${v:,.0f}"

def get_scope_data(df, brands, categories):
    selected = df.copy()
    if brands:
        selected = selected[selected["brand"].isin(brands)]
    if categories:
        selected = selected[selected["category"].isin(categories)]
    if selected.empty:
        return selected, pd.DataFrame()
    daily = selected.groupby("date", as_index=False)[["store", "ad", "organic"]].sum()
    daily["online"] = daily["ad"] + daily["organic"]
    daily["overall"] = daily["store"] + daily["online"]
    return selected, daily.sort_values("date")

def forecast_series(daily, end_date, level=80, model_choice="AutoETS"):
    """Forecast the three base series; derived series are summed from them."""
    last_date = pd.Timestamp(daily["date"].max())
    horizon = max(0, (pd.Timestamp(end_date) - last_date).days)
    if horizon == 0:
        return pd.DataFrame(columns=["date"] + BASE_METRICS + [
            f"{m}_lo" for m in BASE_METRICS
        ] + [f"{m}_hi" for m in BASE_METRICS]), last_date

    # Daily continuous index. Missing dates are filled with 0 in this MVP.
    full_dates = pd.date_range(daily["date"].min(), last_date, freq="D")
    regular = daily.set_index("date").reindex(full_dates, fill_value=0)
    pieces = []
    for metric in BASE_METRICS:
        y = pd.to_numeric(regular[metric], errors="coerce").fillna(0).astype(float).clip(lower=0)
        train = pd.DataFrame({
            "unique_id": metric,
            "ds": full_dates,
            "y": y.values,
        })
        model = AutoETS(season_length=7, model="ZZZ") if model_choice == "AutoETS" else SeasonalNaive(season_length=7)
        model_key = "AutoETS" if model_choice == "AutoETS" else "SeasonalNaive"
        try:
            sf = StatsForecast(models=[model], freq="D", n_jobs=1)
            result = sf.forecast(df=train, h=horizon, level=[level]).reset_index()
            pred_col = model_key
            lo_col, hi_col = f"{model_key}-lo-{level}", f"{model_key}-hi-{level}"
            piece = pd.DataFrame({
                "date": pd.to_datetime(result["ds"]),
                metric: pd.to_numeric(result[pred_col], errors="coerce").clip(lower=0),
                f"{metric}_lo": pd.to_numeric(result[lo_col], errors="coerce").clip(lower=0),
                f"{metric}_hi": pd.to_numeric(result[hi_col], errors="coerce").clip(lower=0),
            })
        except Exception:
            # Safe fallback for short/degenerate series: recent 7-day average with a widening illustrative band.
            recent = y.tail(min(7, len(y))).mean() if len(y) else 0.0
            dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
            preds = [max(0, recent)] * horizon
            los, his = [], []
            for i, value in enumerate(preds, start=1):
                spread = min(0.8, 0.15 + 0.01 * i) * (1.0 if level == 80 else 1.35)
                los.append(max(0, value * (1 - spread)))
                his.append(value * (1 + spread))
            piece = pd.DataFrame({"date": dates, metric: preds, f"{metric}_lo": los, f"{metric}_hi": his})
        pieces.append(piece)
    merged = pieces[0]
    for p in pieces[1:]:
        merged = merged.merge(p, on="date", how="inner")
    merged["online"] = merged["ad"] + merged["organic"]
    merged["overall"] = merged["store"] + merged["online"]
    # Approximate derived bounds by summing component bounds. This is not a calibrated interval.
    merged["online_lo"] = merged["ad_lo"] + merged["organic_lo"]
    merged["online_hi"] = merged["ad_hi"] + merged["organic_hi"]
    merged["overall_lo"] = merged["store_lo"] + merged["online_lo"]
    merged["overall_hi"] = merged["store_hi"] + merged["online_hi"]
    return merged, last_date

# The bundled demo CSV is loaded on the server; no upload or model controls.
model_choice = MODEL_CHOICE
level = PREDICTION_LEVEL
try:
    if not DATA_PATH.is_file():
        raise FileNotFoundError("Bundled demo data is missing: Sales_Sample.csv")
    df, invalid_count, found_cols = prepare_data(DATA_PATH)
except Exception as exc:
    st.error(f"Could not load the bundled demo data: {exc}")
    st.stop()

if StatsForecast is None:
    st.error("StatsForecast is not installed. Install the packages in requirements.txt.")
    st.stop()

max_data_date = pd.Timestamp(df["date"].max()).date()
min_data_date = pd.Timestamp(df["date"].min()).date()

# Persisted sandbox list
sandboxes = load_sandboxes()
if "selected_sandbox_id" not in st.session_state:
    st.session_state.selected_sandbox_id = sandboxes[0]["id"]

def find_selected():
    for item in sandboxes:
        if item["id"] == st.session_state.selected_sandbox_id:
            return item
    return sandboxes[0]

# Sandbox creation and editing are handled in dialogs.
@st.dialog("Sandbox Settings", width="large")
def sandbox_dialog(item_id=None):
    creating = item_id is None
    item = next((x for x in sandboxes if x["id"] == item_id), None) if not creating else None
    if item is None:
        item = {"name": "New Sandbox", "brands": [], "categories": [],
                "start_date": str(min_data_date),
                "end_date": str(max_data_date + timedelta(days=30)), "targets": {}}
    st.caption("Choose the business scope and date range to generate a forecast. All targets are optional.")
    name = st.text_input("Sandbox Name", value=item.get("name", "New Sandbox"))
    brands_all = sorted(x for x in df["brand"].dropna().unique().tolist() if x)
    cats_all = sorted(x for x in df["category"].dropna().unique().tolist() if x)
    left, right = st.columns(2)
    with left:
        brands = st.multiselect("Brand", brands_all, default=[x for x in item.get("brands", []) if x in brands_all])
    with right:
        categories = st.multiselect("Category", cats_all, default=[x for x in item.get("categories", []) if x in cats_all])
    try:
        start_value = date.fromisoformat(item.get("start_date") or str(min_data_date))
    except ValueError:
        start_value = min_data_date
    start_value = max(min_data_date, min(start_value, max_data_date))
    try:
        end_value = date.fromisoformat(item.get("end_date") or str(max_data_date + timedelta(days=30)))
    except ValueError:
        end_value = max_data_date + timedelta(days=30)
    end_value = max(end_value, start_value)
    with left:
        start = st.date_input("Start Date", value=start_value, min_value=min_data_date, max_value=max_data_date)
    with right:
        end = st.date_input("End Date", value=max(end_value, start), min_value=start, max_value=max_data_date + timedelta(days=730))
    st.markdown("**Optional cumulative targets**")
    st.caption("Select a metric to set its cumulative target. Unselected metrics show no target line or target percentages.")
    target_values = {}
    target_cols = st.columns(4)
    for col, (label, key) in zip(target_cols, METRICS.items()):
        with col:
            old = item.get("targets", {}).get(key)
            enabled = st.checkbox("Set target", value=old is not None, key=f"dlg_{item_id or 'new'}_{key}_enabled")
            value = st.number_input(label, min_value=0.0, value=float(old or 0.0), step=1000.0, disabled=not enabled, key=f"dlg_{item_id or 'new'}_{key}_value")
            target_values[key] = value if enabled else None
    saved = st.button("Save and Update Forecast" if not creating else "Create Sandbox", type="primary", use_container_width=True)
    if saved:
        if creating:
            numeric_ids = [int(re.sub(r"\D", "", x["id"]) or 0) for x in sandboxes]
            item["id"] = "sandbox-" + str(max(numeric_ids + [0]) + 1)
            sandboxes.append(item)
        item.update({"name": name.strip() or "Untitled Sandbox", "brands": brands,
                     "categories": categories, "start_date": str(start), "end_date": str(end),
                     "targets": target_values})
        save_sandboxes(sandboxes)
        st.rerun()

# One creation button in the page header; no sidebar controls.
heading, add_area = st.columns([7, 2], vertical_alignment="center")
with heading:
    st.title("Target Sandbox")
    st.caption("Explore sales forecasts, compare scenarios, and evaluate optional targets.")
with add_area:
    if st.button("＋ New Sandbox", type="primary", use_container_width=True):
        sandbox_dialog()
st.caption(f"Demo data through {max_data_date:%b %d, %Y} · {df['sku'].nunique()} SKUs · Forecast: {model_choice} · {level}% nominal interval")
st.divider()

# KPI summary and original-prototype-style sandbox cards
st.divider()

# Data and forecast depend on each Sandbox. Cache computation to avoid repeated work on reruns.
@st.cache_data(show_spinner=False)
def calculate_sandbox_metrics(data, start_iso, end_iso, brands, categories, model, coverage):
    scope, daily_data = get_scope_data(data, list(brands), list(categories))
    if daily_data.empty:
        return None
    start = pd.Timestamp(start_iso).normalize()
    end = pd.Timestamp(end_iso).normalize()
    if end < start:
        return None
    predicted, observed_end = forecast_series(daily_data, end.date(), level=coverage, model_choice=model)
    dates = pd.date_range(start, end, freq="D")
    actual_day = daily_data.set_index("date").reindex(dates)[["store", "ad", "organic"]].fillna(0)
    actual_day["online"] = actual_day["ad"] + actual_day["organic"]
    actual_day["overall"] = actual_day["store"] + actual_day["online"]
    result = pd.DataFrame(index=dates)
    predicted_idx = predicted.set_index("date") if not predicted.empty else pd.DataFrame()
    for key in ["store", "ad", "organic", "online", "overall"]:
        result[f"{key}_actual"] = actual_day[key].where(dates <= observed_end, 0.0)
        result[f"{key}_forecast"] = (predicted_idx[key].reindex(dates).fillna(0.0)
            if not predicted.empty else 0.0)
        result[f"{key}_daily"] = result[f"{key}_actual"].where(dates <= observed_end, result[f"{key}_forecast"])
        for bound in ("lo", "hi"):
            column = f"{key}_{bound}"
            result[column] = (predicted_idx[column].reindex(dates).fillna(0.0)
                if not predicted.empty else 0.0)
    return result, observed_end, len(scope)


@st.cache_data(show_spinner=False)
def calculate_forecast_trend(data, start_iso, end_iso, brands, categories, model, coverage):
    """Five retrospective forecast origins, always predicting the same Sandbox final date.

    Cutoffs use the last five distinct calendar days in the uploaded data range.
    Only observations at or before each cutoff may be used in model fitting.
    """
    _, daily = get_scope_data(data, list(brands), list(categories))
    if daily.empty:
        return []
    start = pd.Timestamp(start_iso).normalize()
    end = pd.Timestamp(end_iso).normalize()
    latest = pd.Timestamp(data["date"].max()).normalize()
    snapshots = []
    for offset in range(4, -1, -1):
        cutoff = latest - pd.Timedelta(days=offset)
        history = daily.loc[daily["date"] <= cutoff].copy()
        if history.empty or len(history) < 14:
            continue
        # Keep the model training window identical to the main forecast, but
        # hide all records after the simulated observation cutoff.
        forecast, _ = forecast_series(history, end.date(), level=coverage, model_choice=model)
        known = history.loc[(history["date"] >= start) & (history["date"] <= end)]
        future = forecast.loc[(forecast["date"] >= start) & (forecast["date"] <= end)]
        values = {}
        for metric in METRICS.values():
            values[metric] = float(known[metric].sum()) + float(future[metric].sum())
        snapshots.append({"date": cutoff.date().isoformat(), "values": values})
    return snapshots


def trend_html(snapshots, metric):
    """Five historical forecast totals; no subjective trend classification."""
    valid = [(s["date"], float(s["values"][metric])) for s in snapshots]
    if len(valid) < 2:
        return '<span class="trend-unavailable" title="Not enough forecast snapshots">—</span>'
    values = [value for _, value in valid]
    color = "#4c88c6"
    lo, hi = min(values), max(values)
    # Keep very small changes visually modest rather than exaggerating them.
    padding = max((hi - lo) * 0.20, max(abs(sum(values) / len(values)), 1) * 0.005)
    lo -= padding
    hi += padding
    coords = [(4 + i * 68 / (len(values) - 1), 22 - (value - lo) / (hi - lo) * 18)
              for i, value in enumerate(values)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.8 if i == len(coords)-1 else 1.8}" fill="{color}"/>'
                   for i, (x, y) in enumerate(coords))
    details = "".join(f'<div class="trend-detail"><span>{html.escape(day)}</span><b>{money(value)}</b></div>'
                      for day, value in valid)
    return (f'<span class="forecast-trend" tabindex="0" aria-label="Last {len(valid)} simulated forecast totals">'
            f'<span class="trend-compact"><svg viewBox="0 0 76 27" width="76" height="27" aria-hidden="true">'
            f'<polyline points="{path}" stroke="{color}" stroke-width="1.8" fill="none" '
            f'stroke-linejoin="round" stroke-linecap="round"/>{dots}</svg></span>'
            f'<span class="trend-tooltip"><strong>Last {len(valid)} Forecasts · Simulated</strong>'
            f'{details}<small>Same Sandbox end date · Different historical data cutoffs</small>'
            f'</span></span>')


def pct_of_target(value, target):
    if target is None:
        return ""
    if float(target) == 0:
        return "(N/A: target is zero)"
    return f"({value / float(target) * 100:,.1f}%)"


def compact_money(value):
    return f"${value:,.0f}"


def draw_sandbox_card(item):
    label = item.get("name", "Untitled Sandbox")
    brands = tuple(item.get("brands") or ())
    cats = tuple(item.get("categories") or ())
    start_iso = item.get("start_date") or str(min_data_date)
    end_iso = item.get("end_date") or str(max_data_date + timedelta(days=30))
    if start_iso > end_iso:
        st.warning(f"{label} has an end date before its start date.")
        return
    computed = calculate_sandbox_metrics(df, start_iso, end_iso, brands, cats, model_choice, level)
    if computed is None:
        st.warning(f"{label} has no sales data for the selected scope.")
        return
    series, observed_end, record_count = computed
    if series.empty:
        return
    snapshots = calculate_forecast_trend(df, start_iso, end_iso, brands, cats, model_choice, level)
    targets_for_card = item.get("targets") or {}
    days_total = len(series)
    days_past = int((series.index <= observed_end).sum())
    scope_text = f"Brand: {', '.join(brands) if brands else 'All'}; Category: {', '.join(cats) if cats else 'All'}"
    with st.container(border=True):
        header, edit_area = st.columns([9, 1], vertical_alignment="center")
        with header:
            st.markdown(f"#### {label}")
        with edit_area:
            if st.button("✏️ Edit", key=f"edit_{item['id']}", use_container_width=True):
                sandbox_dialog(item["id"])
        st.caption(f"Scope: {scope_text}")
        st.caption(f"Time Range: {start_iso} - {end_iso}　 {days_total} Days Total　 {days_past / days_total * 100:.0f}% Past")
        columns = st.columns(4, gap="small")
        for col, (display, key) in zip(columns, METRICS.items()):
            with col:
                st.markdown(f"**{display}**")
                goal = targets_for_card.get(key)
                goal = None if goal is None else float(goal)
                actual_total = float(series[f"{key}_actual"].sum())
                final_total = float(series[f"{key}_daily"].sum())
                # These are approximate cumulative envelopes; not calibrated interval bounds.
                lower_total = actual_total + float(series[f"{key}_lo"].sum())
                upper_total = actual_total + float(series[f"{key}_hi"].sum())
                st.markdown(f"<div class='metric-line'>Target: <b>{compact_money(goal) if goal is not None else '—'}</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-line'>Actual: <b>{compact_money(actual_total)}</b> {pct_of_target(actual_total, goal)}</div>", unsafe_allow_html=True)
                forecast_pct = pct_of_target(final_total, goal)
                range_pct = (f"{lower_total / goal * 100:,.1f}% – {upper_total / goal * 100:,.1f}%"
                             if goal is not None and goal > 0 else "")
                st.markdown(
                    f"<div class='metric-line forecast-final-row'><span class='forecast-final-value'>Forecast Final: <b>{compact_money(final_total)}</b> "
                    f"<strong class='target-pct'>{forecast_pct}</strong></span>{trend_html(snapshots, key)}</div>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<div class='metric-line'>Range: <b>{compact_money(lower_total)} – {compact_money(upper_total)}</b> "
                    f"<strong class='target-pct'>{'(' + range_pct + ')' if range_pct else ''}</strong></div>",
                    unsafe_allow_html=True)
                fig = go.Figure()
                actual_dates = series.index[series.index <= observed_end]
                future_dates = series.index[series.index > observed_end]
                actual_cumulative = series[f"{key}_actual"].cumsum()
                fig.add_trace(go.Scatter(x=actual_dates, y=actual_cumulative.loc[actual_dates], mode="lines", name="Actual", line=dict(color="#6484a5", width=2)))
                if len(future_dates):
                    future_point = actual_total + series.loc[future_dates, f"{key}_forecast"].cumsum()
                    future_low = actual_total + series.loc[future_dates, f"{key}_lo"].cumsum()
                    future_high = actual_total + series.loc[future_dates, f"{key}_hi"].cumsum()
                    xs = [observed_end] + list(future_dates) if observed_end in series.index else list(future_dates)
                    start_value = [actual_total] if observed_end in series.index else []
                    fig.add_trace(go.Scatter(x=xs, y=start_value + list(future_high), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=xs, y=start_value + list(future_low), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(110,110,110,.12)", showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=xs, y=start_value + list(future_point), mode="lines", name="Forecast", line=dict(color="#84b6da", width=2)))
                # Subtle boundary between observed sales and the forecast period.
                # Only draw it when the boundary falls inside this Sandbox time window.
                if len(actual_dates) and len(future_dates):
                    fig.add_vline(
                        x=observed_end.timestamp() * 1000,
                        line_color="rgba(105,115,130,0.40)",
                        line_width=1,
                        line_dash="dot",
                    )
                if goal is not None:
                    fig.add_hline(y=goal, line_color="#e85c50", line_width=1.6)
                fig.update_layout(height=225, margin=dict(l=0, r=3, t=5, b=0), showlegend=False,
                    xaxis=dict(title=None, tickfont=dict(size=9), showgrid=False),
                    yaxis=dict(title=None, tickfont=dict(size=9), gridcolor="rgba(140,140,140,.14)"),
                    hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{item['id']}_{key}")

st.markdown("""<style>
.metric-line {font-size: 12px; min-height: 28px; padding: 5px 2px; border-bottom: 1px solid rgba(120,120,120,.18); overflow-wrap:anywhere;}
.metric-line b {font-weight: 650;}
.forecast-final-row {display:flex; align-items:center; justify-content:space-between; gap:3px; position:relative;}
.forecast-final-value {flex:1; min-width:0;}
.forecast-trend {display:inline-flex; position:relative; flex:0 0 76px; width:76px; cursor:help; outline-offset:2px;}
.trend-compact {display:flex; align-items:center; justify-content:flex-end; line-height:1;}
.trend-tooltip {display:none; position:absolute; z-index:999; top:100%; right:0; width:230px; padding:11px 12px; background:#fff; color:#293241; border:1px solid #d5dbe3; border-radius:8px; box-shadow:0 7px 24px rgba(0,0,0,.14); font-size:11px; overflow-wrap:normal;}
.forecast-trend:hover .trend-tooltip, .forecast-trend:focus .trend-tooltip, .forecast-trend:focus-within .trend-tooltip {display:block;}
.trend-tooltip>strong {display:block; margin-bottom:4px;}
.trend-detail {display:flex; justify-content:space-between; gap:10px; padding:3px 0; border-bottom:1px solid #edf0f4;}
.trend-tooltip small {display:block; margin-top:7px; color:#687587; line-height:1.5;}
.trend-unavailable {color:#8b929a; font-size:11px;}

.target-pct {display: inline-block; color: #b45309; font-weight: 750; margin-left: 5px; font-size: 13px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 9px;}
</style>""", unsafe_allow_html=True)

# Render every Sandbox card in its saved order.
for item in sandboxes:
    draw_sandbox_card(item)

st.caption("Prediction interval note: The cumulative bands approximate sums of daily prediction bounds. Online and Overall bounds are sums of the base metric bounds. These are not backtested or calibrated cumulative 80%/95% prediction intervals.")
st.caption("Forecast Trend: Five simulated forecasts use the last five calendar cutoffs in the uploaded data, all predicting the same Sandbox end date. Hover over a sparkline beside Forecast Final to see the five values. No trend classification is applied.")

