# crypto_dashboard.py
import time
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import timedelta, timezone

st.set_page_config(page_title="Live Crypto Analytics", layout="wide")

# -----------------------------
# UI
# -----------------------------
st.title("📈 Live Crypto Analytics (5-Hour View, Outliers, FX, Forecast)")
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    coin = st.selectbox("Cryptocurrency", ["bitcoin", "ethereum", "dogecoin", "solana", "cardano"], index=0)
with c2:
    refresh_sec = st.slider("Auto-refresh every (seconds)", 15, 180, 60)
with c3:
    auto_refresh = st.checkbox("Auto-refresh", value=False)

# A safe rerun helper (supports older/newer Streamlit)
def do_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

HEADERS = {"User-Agent": "Mozilla/5.0"}

# -----------------------------
# Data fetchers (cached)
# -----------------------------
@st.cache_data(ttl=55)
def fetch_market_chart(coin_id: str, days: int = 2) -> pd.DataFrame:
    """
    Free-plan compatible: don't pass interval=hourly.
    Returns minute-ish granularity for last `days`.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"API Error {r.status_code}: {r.text[:200]}")
    data = r.json()
    if "prices" not in data:
        raise RuntimeError(f"Unexpected API response (no 'prices'): {str(data)[:200]}")
    df = pd.DataFrame(data["prices"], columns=["ts_ms", "price_usd"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert("UTC")
    df = df[["ts", "price_usd"]].sort_values("ts").reset_index(drop=True)
    return df

@st.cache_data(ttl=55)
def fetch_spot_multi(coin_id: str) -> Dict[str, float]:
    """Spot price in USD, INR, GBP."""
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin_id, "vs_currencies": "usd,inr,gbp"}
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"API Error {r.status_code}: {r.text[:160]}")
    d = r.json().get(coin_id, {})
    return {"usd": float(d.get("usd", np.nan)),
            "inr": float(d.get("inr", np.nan)),
            "gbp": float(d.get("gbp", np.nan))}

# -----------------------------
# Window builders & analytics
# -----------------------------
def last_5h_and_yesterday(df_48h: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    now_utc = pd.Timestamp.now(tz=timezone.utc)
    cur_start = now_utc - pd.Timedelta(hours=5)

    cur = df_48h[(df_48h["ts"] >= cur_start) & (df_48h["ts"] <= now_utc)].copy()

    # Yesterday’s same 5h window (24h earlier)
    y_start = cur_start - pd.Timedelta(hours=24)
    y_end = now_utc - pd.Timedelta(hours=24)
    yest = df_48h[(df_48h["ts"] >= y_start) & (df_48h["ts"] <= y_end)].copy()

    # Align by window index so the two series overlay nicely in a comparison chart
    cur = cur.reset_index(drop=True)
    yest = yest.reset_index(drop=True)
    cur["minute_idx"] = np.arange(len(cur))
    yest["minute_idx"] = np.arange(len(yest))
    return cur, yest

def detect_outliers_z(df: pd.DataFrame, col="price_usd", z_thresh: float = 3.0) -> pd.DataFrame:
    """Flag outliers using z-score of percentage returns."""
    d = df.copy()
    s = d[col].astype(float)
    rets = s.pct_change().fillna(0.0)
    z = (rets - rets.mean()) / (rets.std(ddof=0) + 1e-12)
    d["is_outlier"] = z.abs() > z_thresh
    return d

def scale_multi_currency(df: pd.DataFrame, fx: Dict[str, float]) -> pd.DataFrame:
    """
    Convert USD series into INR/GBP using spot ratios (lightweight approximation).
    """
    out = df.copy()
    if fx.get("usd") and fx.get("inr"):
        out["price_inr"] = out["price_usd"] * (fx["inr"] / fx["usd"])
    else:
        out["price_inr"] = np.nan
    if fx.get("usd") and fx.get("gbp"):
        out["price_gbp"] = out["price_usd"] * (fx["gbp"] / fx["usd"])
    else:
        out["price_gbp"] = np.nan
    return out

def forecast_next_5h_linear(df_5h: pd.DataFrame) -> pd.DataFrame:
    """Simple linear regression on last 5h vs time (minutes) → predict next 300 mins."""
    if len(df_5h) < 10:
        return pd.DataFrame(columns=["ts", "yhat_usd"])
    d = df_5h.copy()
    t0 = d["ts"].min()
    d["t_min"] = (d["ts"] - t0).dt.total_seconds() / 60.0
    X = d[["t_min"]].values
    y = d["price_usd"].values
    lr = LinearRegression()
    lr.fit(X, y)
    last_min = d["t_min"].max()
    future_minutes = np.arange(last_min + 1, last_min + 300 + 1)  # next 300 mins
    yhat = lr.predict(future_minutes.reshape(-1, 1))
    future_ts = [d["ts"].max() + timedelta(minutes=i) for i in range(1, 301)]
    return pd.DataFrame({"ts": future_ts, "yhat_usd": yhat})

# -----------------------------
# Fetch & guard
# -----------------------------
try:
    df_48h = fetch_market_chart(coin, days=2)
    fx = fetch_spot_multi(coin)
except Exception as e:
    st.error(str(e))
    st.stop()

# Use only last 24h just to keep things tight
cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(hours=24)
df_24h = df_48h[df_48h["ts"] >= cutoff].copy()
if df_24h.empty:
    st.warning("No recent data returned. Try again later.")
    st.stop()

cur5, yest5 = last_5h_and_yesterday(df_24h)
cur5 = detect_outliers_z(cur5, "price_usd", z_thresh=3.0)
cur5_fx = scale_multi_currency(cur5, fx)
forecast5 = forecast_next_5h_linear(cur5)

# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
spot_usd = fx.get("usd", np.nan)
spot_inr = fx.get("inr", np.nan)
spot_gbp = fx.get("gbp", np.nan)

# Yesterday ref = last point of yesterday window (if present)
y_ref = yest5["price_usd"].iloc[-1] if not yest5.empty else np.nan
chg_pct_vs_y = (spot_usd - y_ref) / y_ref * 100 if (y_ref and not np.isnan(y_ref)) else np.nan

k1.metric(f"{coin.capitalize()} (USD)", f"${spot_usd:,.2f}" if spot_usd==spot_usd else "—")
k2.metric(f"{coin.capitalize()} (INR)", f"₹{spot_inr:,.2f}" if spot_inr==spot_inr else "—")
k3.metric(f"{coin.capitalize()} (GBP)", f"£{spot_gbp:,.2f}" if spot_gbp==spot_gbp else "—")
k4.metric("Δ vs yesterday (same time)", f"{chg_pct_vs_y:,.2f}%" if chg_pct_vs_y==chg_pct_vs_y else "—")

st.caption(f"Last updated: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S %Z')}")

# -----------------------------
# Chart 1 — Last 5 hours (USD) with outliers
# -----------------------------
st.subheader("Last 5 hours (USD) — outliers highlighted")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=cur5["ts"], y=cur5["price_usd"], mode="lines", name="Price (USD)"))
if cur5["is_outlier"].any():
    o = cur5[cur5["is_outlier"]]
    fig1.add_trace(go.Scatter(x=o["ts"], y=o["price_usd"], mode="markers", name="Outliers",
                              marker=dict(size=8, symbol="circle-open")))
fig1.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=360)
st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# Chart 2 — Today vs Yesterday (aligned by minute index)
# -----------------------------
st.subheader("Compare with yesterday (same 5-hour window)")
fig2 = go.Figure()
if not yest5.empty:
    fig2.add_trace(go.Scatter(x=yest5["minute_idx"], y=yest5["price_usd"], mode="lines", name="Yesterday (USD)"))
fig2.add_trace(go.Scatter(x=cur5["minute_idx"], y=cur5["price_usd"], mode="lines", name="Today (USD)"))
fig2.update_layout(xaxis_title="Minute in window (0–300)", margin=dict(l=10, r=10, t=30, b=10), height=360)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Chart 3 — Multi-currency (USD / INR / GBP)
# -----------------------------
st.subheader("Multi-currency (last 5 hours)")
cur_multi = pd.DataFrame({
    "ts": cur5_fx["ts"],
    "USD": cur5_fx["price_usd"],
    "INR": cur5_fx.get("price_inr", np.nan),
    "GBP": cur5_fx.get("price_gbp", np.nan),
})
fig3 = go.Figure()
for col in ["USD", "INR", "GBP"]:
    if cur_multi[col].notna().any():
        fig3.add_trace(go.Scatter(x=cur_multi["ts"], y=cur_multi[col], mode="lines", name=col))
fig3.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=360)
st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# Chart 4 — Forecast next 5 hours (simple linear trend)
# -----------------------------
st.subheader("Prediction — next 5 hours (USD, linear trend)")
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=cur5["ts"], y=cur5["price_usd"], mode="lines", name="History (5h)"))
if not forecast5.empty:
    fig4.add_trace(go.Scatter(x=forecast5["ts"], y=forecast5["yhat_usd"], mode="lines",
                              name="Forecast (5h)", line=dict(dash="dash")))
    st.info(f"Forecasted price in 5 hours: **${forecast5['yhat_usd'].iloc[-1]:,.2f}**")
else:
    st.warning("Not enough recent points to forecast.")
fig4.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=360)
st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# Controls
# -----------------------------
cA, cB = st.columns([1, 1])
with cA:
    if st.button("Refresh now"):
        st.cache_data.clear()
        do_rerun()
with cB:
    st.caption("Tip: enable Auto-refresh for hands-free updates.")

# Optional auto-refresh
if auto_refresh:
    time.sleep(refresh_sec)
    st.cache_data.clear()
    do_rerun()
