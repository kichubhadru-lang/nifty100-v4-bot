"""
Conservative price-only research backtest.

It does not claim point-in-time fundamental or news accuracy.
Current Nifty 100 membership causes survivorship bias.
Daily bars cannot reproduce the live first-15-minute breakout precisely.
"""
from __future__ import annotations
import argparse
import pandas as pd
import yfinance as yf
import config
from data_provider import nifty100_symbols
from indicators import rsi
from utils import flatten_yf_columns


def test_symbol(symbol: str, start: str, end: str | None, target: float) -> list[dict]:
    df = yf.download(
        symbol, start=start, end=end, auto_adjust=True,
        progress=False, threads=False
    )
    df = flatten_yf_columns(df).dropna()
    if len(df) < 260:
        return []

    df["change"] = df["Close"].pct_change() * 100
    df["dma200"] = df["Close"].rolling(200).mean()
    df["rsi"] = rsi(df["Close"])
    df["avg_value"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    trades = []
    for i in range(252, len(df) - 1):
        row, nxt = df.iloc[i], df.iloc[i + 1]
        distance = (row["Close"] / row["dma200"] - 1) * 100
        new_low = row["Low"] <= df["Low"].iloc[i-251:i+1].min() * 1.002
        eligible = (
            config.MIN_DROP_PCT <= row["change"] <= config.MAX_DROP_PCT
            and row["Close"] >= config.MIN_PRICE
            and row["avg_value"] >= config.MIN_AVG_VALUE_20D
            and config.MIN_DISTANCE_200DMA <= distance <= config.MAX_DISTANCE_200DMA
            and config.MIN_RSI <= row["rsi"] <= config.MAX_RSI
            and row["volume_ratio"] >= config.MIN_VOLUME_RATIO
            and not new_low
        )
        if not eligible:
            continue

        gap = (nxt["Open"] / row["Close"] - 1) * 100
        if not config.MIN_GAP_PCT <= gap <= config.MAX_GAP_PCT:
            continue

        # Daily-data proxy: bullish day required; entry at open is optimistic compared with live breakout.
        if config.REQUIRE_BULLISH_FIRST_15M and nxt["Close"] <= nxt["Open"]:
            continue

        entry = float(nxt["Open"])
        stop = entry * (1 - config.STOP_LOSS_PCT / 100)
        target_price = entry * (1 + target / 100)

        if nxt["Low"] <= stop and nxt["High"] >= target_price:
            exit_price, reason = stop, "both_assume_stop"
        elif nxt["Low"] <= stop:
            exit_price, reason = stop, "stop"
        elif nxt["High"] >= target_price:
            exit_price, reason = target_price, "target"
        else:
            exit_price, reason = float(nxt["Close"]), "time"

        trades.append({
            "symbol": symbol,
            "signal_date": str(df.index[i].date()),
            "entry_date": str(df.index[i+1].date()),
            "return_pct": (exit_price / entry - 1) * 100,
            "reason": reason,
        })
    return trades


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--target", type=float, default=2.0)
    args = p.parse_args()

    trades = []
    for symbol in nifty100_symbols():
        try:
            found = test_symbol(symbol, args.start, args.end, args.target)
            trades.extend(found)
            print(symbol, len(found))
        except Exception as exc:
            print(symbol, exc)

    out = pd.DataFrame(trades)
    out.to_csv(f"backtest_target_{args.target}.csv", index=False)
    if out.empty:
        print("No trades.")
        return

    equity = (1 + out["return_pct"] / 100).cumprod()
    drawdown = equity / equity.cummax() - 1
    print("Trades:", len(out))
    print("Win rate:", round((out["return_pct"] > 0).mean() * 100, 2))
    print("Average/trade:", round(out["return_pct"].mean(), 3))
    print("Compounded:", round((equity.iloc[-1] - 1) * 100, 2))
    print("Max drawdown:", round(drawdown.min() * 100, 2))
    print(out["reason"].value_counts())


if __name__ == "__main__":
    main()
