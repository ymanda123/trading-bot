"""Central configuration for the regime-switching trading bot.

All values can be overridden via environment variables (see .env.example)
except the risk-management constants, which encode the fixed rules from
SPEC.md and are intentionally not user-configurable via env vars.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- Exchange (CCXT / Binance Testnet) ---------------------------------
    EXCHANGE_ID = "binance"
    API_KEY = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    TESTNET = True
    SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
    TIMEFRAME = os.getenv("TIMEFRAME", "1h")

    # --- Tradable assets (fixed set, selectable via server.py's /start) --------
    # "crypto" symbols route through the ccxt/Binance -> Coinbase -> Kraken
    # chain in bot.py; "stock" and "commodity" symbols have no exchange
    # account behind them at all, so they route through Yahoo Finance's
    # public chart API instead (see bot._fetch_ohlcv_yahoo). Either way this
    # stays paper-trading only -- nothing here ever places a real order.
    SUPPORTED_ASSETS = {
        "BTC/USDT": {"label": "Bitcoin (BTC/USDT)", "kind": "crypto", "yahoo_symbol": "BTC-USD"},
        "ETH/USDT": {"label": "Ethereum (ETH/USDT)", "kind": "crypto", "yahoo_symbol": "ETH-USD"},
        "XAU/USD": {"label": "Gold futures (XAU/USD)", "kind": "commodity", "yahoo_symbol": "GC=F"},
        "WTI/USD": {"label": "Crude oil futures (WTI)", "kind": "commodity", "yahoo_symbol": "CL=F"},
        "AAPL": {"label": "Apple (AAPL)", "kind": "stock", "yahoo_symbol": "AAPL"},
        "TSLA": {"label": "Tesla (TSLA)", "kind": "stock", "yahoo_symbol": "TSLA"},
    }

    # --- Capital & circuit breaker ------------------------------------------
    INITIAL_BALANCE = 200.0
    CIRCUIT_BREAKER_DRAWDOWN_PCT = 0.20
    CIRCUIT_BREAKER_BALANCE = INITIAL_BALANCE * (1 - CIRCUIT_BREAKER_DRAWDOWN_PCT)  # $160

    # --- Progressive position sizing ----------------------------------------
    # consecutive_losses -> (size_pct_of_balance, atr_stop_multiple)
    POSITION_SIZING_TIERS = {
        0: (1.00, 2.0),  # Trade 1: 100% size, 2.0x ATR stop
        1: (0.50, 1.5),  # Trade 2: 50% size, 1.5x ATR stop
        2: (0.25, 1.0),  # Trade 3+: 25% size, 1.0x ATR stop
    }
    MAX_SIZE_TIER_LOSSES = max(POSITION_SIZING_TIERS)

    # --- Cool-off ------------------------------------------------------------
    CONSECUTIVE_LOSS_COOLOFF_THRESHOLD = 3
    COOLOFF_MINUTES = 15

    # --- ATR / regime classification ------------------------------------------
    ATR_PERIOD = 14
    LOW_VOL_ATR_PCT_THRESHOLD = 0.01   # ATR/price <= 1%  -> low vol / range-bound
    HIGH_VOL_ATR_PCT_THRESHOLD = 0.03  # ATR/price >= 3%  -> high vol / spike -> standby
    # Between the two thresholds -> moderate vol / trending

    # --- Trend-following (moderate volatility regime) --------------------------
    FAST_MA_PERIOD = 10
    SLOW_MA_PERIOD = 30

    # --- Grid / mean-reversion (low volatility regime) --------------------------
    GRID_LOOKBACK = 20  # bars used to define the support/resistance range

    # --- Friction ------------------------------------------------------------
    TAKER_FEE_PCT = 0.00075  # 0.075%
    SLIPPAGE_PCT = 0.0005    # 0.05%

    # --- AI strategy proposal + backtest gate ---------------------------------
    AI_HISTORY_CANDLES = 150  # bars of history given to the AI and used to backtest its proposed strategy
    BACKTEST_MIN_TRADES = 3   # minimum trades in the backtest window before a strategy's edge is trusted
