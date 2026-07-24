# Nifty 100 V4 Mean-Reversion Telegram Bot

## What V4 actually does

V4 is a paper-trading research system, not an automatic broker-order bot.

1. Scans the official Nifty 100 universe after market close.
2. Applies market, liquidity, fundamental, technical and basic news guardrails.
3. Scores candidates using five components.
4. Confirms only next-day first-15-minute breakout setups.
5. Selects a maximum of one paper trade.
6. Records the outcome at 3:15 PM.
7. Produces weekly performance reports.
8. Adjusts scoring weights monthly, only after at least 30 closed trades.
9. Limits each adaptive weight to ±20% of its original value.

## Why the learning is restricted

Unrestricted self-learning on a small trading sample is overfitting, not intelligence.
The adaptive module uses shrinkage, a rolling 100-trade window, a low learning rate,
minimum bucket sizes and hard weight bounds.

## Strategy

### Market regime
- Nifty daily fall must be better than -1%.
- Nifty must be above its 50-DMA.
- India VIX should be below 20 when VIX data is available.

### Candidate
- Official Nifty 100 constituent.
- Daily decline between -2.5% and -6%.
- Price at least ₹100.
- Average 20-day traded value at least ₹50 crore.
- Market cap at least ₹20,000 crore.
- ROE at least 15%.
- Yahoo debt/equity no higher than 50 (approximately 0.50 under Yahoo's convention).
- Positive profit margin.
- Between -5% and +20% relative to 200-DMA.
- RSI between 35 and 52.
- Volume at least 1.2× its 20-day average.
- Not at a new 52-week low.
- No matched severe-news keyword.

### Confirmation
- Opening gap between -1.5% and +2%.
- First 15-minute candle bullish.
- Current price above first 15-minute high.
- Highest-scoring confirmed candidate only.

### Paper exit
- Stop: -1.25%.
- Target 1: +2%.
- Target 2: +3%.
- Time exit at 3:15 PM.
- If daily/intraday ordering is ambiguous, the tracker assumes the stop occurred first.

## Installation from iPhone

1. Create a new GitHub repository.
2. Upload the ZIP contents, preserving `.github/workflows/v4.yml`.
3. In repository **Settings → Secrets and variables → Actions**, add:
   - `TELEGRAM_TOKEN`
   - `TELEGRAM_CHAT_ID`
4. Open **Actions → Nifty 100 V4 Bot → Run workflow**.
5. Run `scan` manually.
6. Check the Telegram response and GitHub Actions log.
7. Use paper signals only until enough outcomes exist.

## Manual commands

```bash
python main.py scan
python main.py confirm
python main.py close
python main.py report
python main.py learn
```

## Backtests

```bash
python backtest.py --start 2021-01-01 --target 2
python backtest.py --start 2021-01-01 --target 3
```

The included backtest is intentionally conservative but incomplete:

- It cannot reproduce first-15-minute entries from daily history.
- It does not use historical point-in-time fundamentals.
- It does not use historical point-in-time index membership.
- It does not reconstruct historical news.
- Therefore it must not be presented as proof of live profitability.

## Data limitations

`yfinance` is a convenient research interface to Yahoo Finance public data. It is not
an exchange-grade or broker-grade feed. Intraday data can be delayed, unavailable or
revised. The news keyword filter is weak and cannot reliably understand corporate events.

For real execution, replace the data provider with a licensed real-time feed and connect
a broker only after a separately audited paper-trading record.
