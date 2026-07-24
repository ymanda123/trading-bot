# Trading Bot Spec

Regime-switching crypto trading bot on **Binance Testnet** (paper trading) via
**CCXT**. Not financial advice; testnet only.

## Files

| File | Purpose |
|---|---|
| `config.py` | All tunable constants and env-driven settings |
| `risk_manager.py` | Circuit breaker, position sizing, cool-off, friction |
| `strategy.py` | ATR regime classification + per-regime signal logic |
| `bot.py` | CCXT integration and the live trade loop |
| `app.py` | Streamlit + Plotly dashboard |
| `test_risk_manager.py` | Pytest coverage for every risk rule below |

## Regime Classification (14-period ATR)

ATR is computed over 14 bars and expressed as a percentage of the current
close (`ATR / price`):

| Regime | Condition | Behavior |
|---|---|---|
| Low Volatility / Range-Bound | ATR% <= 1% | Grid / mean-reversion: buy at support, sell at resistance (20-bar rolling low/high) |
| Moderate Volatility / Trending | 1% < ATR% < 3% | Multi-day trend-following: fast MA(10) / slow MA(30) crossover |
| High Volatility / Spikes | ATR% >= 3% | Cash / standby — no new entries |

## Risk Management Guardrails

**Circuit breaker.** Initial balance $200. Trading halts globally once balance
falls to $160 (20% max drawdown).

**Progressive position sizing**, keyed to the current consecutive-loss streak:

| Trade | Size | ATR stop multiple |
|---|---|---|
| 1st (streak = 0) | 100% of balance | 2.0x ATR |
| 2nd (after 1 loss) | 50% of balance | 1.5x ATR |
| 3rd+ (after 2+ losses) | 25% of balance | 1.0x ATR |

**Cool-off.** 3 consecutive losses triggers a mandatory 15-minute pause on new
entries. The loss streak resets to 0 once the cool-off is triggered, so
trading resumes at full (100%) size afterward.

## Friction Accounting

Every simulated fill (entry and exit) deducts:
- Taker fee: 0.075%
- Estimated slippage: 0.05%

applied to that fill's notional value, on top of the underlying price move.

## Running

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your Binance Testnet API key/secret
pytest                 # verify risk rules
python bot.py           # run the live paper-trading loop
streamlit run app.py    # dashboard
```
