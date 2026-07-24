"""Streamlit dashboard for monitoring the trading bot: live candlestick
chart with regime/signal, balance curve, and risk-manager state."""

import plotly.graph_objects as go
import streamlit as st

from bot import build_exchange, fetch_ohlcv_df
from config import Config
from risk_manager import RiskManager
from strategy import decide

st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title("Regime-Switching Trading Bot")

if "risk_manager" not in st.session_state:
    st.session_state.risk_manager = RiskManager()
if "balance_history" not in st.session_state:
    st.session_state.balance_history = [Config.INITIAL_BALANCE]

rm: RiskManager = st.session_state.risk_manager
st.session_state.balance_history.append(rm.balance)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Balance", f"${rm.balance:,.2f}")
col2.metric(
    "Circuit Breaker",
    "TRIPPED" if rm.circuit_breaker_tripped else "OK",
    delta=f"floor ${Config.CIRCUIT_BREAKER_BALANCE:,.2f}",
    delta_color="off",
)
col3.metric("Consecutive Losses", rm.consecutive_losses)
col4.metric("In Cool-off", "YES" if rm.is_in_cooloff() else "NO")

try:
    exchange = build_exchange()
    df = fetch_ohlcv_df(exchange)
    decision = decide(df)

    st.subheader(f"{Config.SYMBOL} — Regime: {decision.regime.value}")
    st.caption(decision.reason)

    price_fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=Config.SYMBOL,
            )
        ]
    )
    price_fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(price_fig, use_container_width=True)

    balance_fig = go.Figure(
        data=[go.Scatter(y=st.session_state.balance_history, mode="lines+markers", name="Balance")]
    )
    balance_fig.add_hline(
        y=Config.CIRCUIT_BREAKER_BALANCE,
        line_dash="dash",
        line_color="red",
        annotation_text="Circuit Breaker ($160)",
    )
    balance_fig.update_layout(title="Balance Over Time", height=350)
    st.plotly_chart(balance_fig, use_container_width=True)

    size_pct, atr_mult = rm.get_position_sizing()
    st.write(f"Next trade sizing: **{size_pct * 100:.0f}%** of balance, stop = **{atr_mult}x ATR**")

except Exception as exc:
    st.error(f"Could not fetch live data from Binance Testnet: {exc}")
    st.info("Check that BINANCE_API_KEY / BINANCE_API_SECRET are set in your .env file.")
