"""
Conservative price-only research backtest.

Limitations:
- Uses current Nifty 100 membership, creating survivorship bias.
- Does not use point-in-time fundamentals or historical news.
- Daily candles cannot reproduce the live first-15-minute breakout precisely.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

import config
from data_provider import nifty100_symbols
from indicators import rsi
from utils import flatten_yf_columns


MAX_WORKERS = 8
DOWNLOAD_TIMEOUT = 15


def download_symbol(
    symbol: str,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Download one symbol with a timeout."""

    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=DOWNLOAD_TIMEOUT,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    return flatten_yf_columns(df).dropna()


def test_symbol(
    symbol: str,
    start: str,
    end: str | None,
    target: float,
) -> list[dict]:
    df = download_symbol(symbol, start, end)

    if len(df) < 260:
        return []

    df["change"] = df["Close"].pct_change() * 100
    df["dma200"] = df["Close"].rolling(200).mean()
    df["rsi"] = rsi(df["Close"])
    df["avg_value"] = (
        df["Close"] * df["Volume"]
    ).rolling(20).mean()

    average_volume = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / average_volume

    trades: list[dict] = []

    for i in range(252, len(df) - 1):
        row = df.iloc[i]
        next_day = df.iloc[i + 1]

        required_values = [
            row["Close"],
            row["change"],
            row["dma200"],
            row["rsi"],
            row["avg_value"],
            row["volume_ratio"],
            next_day["Open"],
            next_day["High"],
            next_day["Low"],
            next_day["Close"],
        ]

        if any(pd.isna(value) for value in required_values):
            continue

        distance_200dma = (
            row["Close"] / row["dma200"] - 1
        ) * 100

        previous_252_day_low = df["Low"].iloc[
            i - 251 : i + 1
        ].min()

        new_52_week_low = (
            row["Low"] <= previous_252_day_low * 1.002
        )

        eligible = (
            config.MIN_DROP_PCT
            <= row["change"]
            <= config.MAX_DROP_PCT
            and row["Close"] >= config.MIN_PRICE
            and row["avg_value"]
            >= config.MIN_AVG_VALUE_20D
            and config.MIN_DISTANCE_200DMA
            <= distance_200dma
            <= config.MAX_DISTANCE_200DMA
            and config.MIN_RSI
            <= row["rsi"]
            <= config.MAX_RSI
            and row["volume_ratio"]
            >= config.MIN_VOLUME_RATIO
            and not new_52_week_low
        )

        if not eligible:
            continue

        gap_pct = (
            next_day["Open"] / row["Close"] - 1
        ) * 100

        if not (
            config.MIN_GAP_PCT
            <= gap_pct
            <= config.MAX_GAP_PCT
        ):
            continue

        # Daily-candle proxy for the live first-15-minute condition.
        if (
            config.REQUIRE_BULLISH_FIRST_15M
            and next_day["Close"] <= next_day["Open"]
        ):
            continue

        entry_price = float(next_day["Open"])

        stop_price = entry_price * (
            1 - config.STOP_LOSS_PCT / 100
        )

        target_price = entry_price * (
            1 + target / 100
        )

        day_low = float(next_day["Low"])
        day_high = float(next_day["High"])

        # With daily candles, intraday order is unknown.
        # Conservatively assume the stop occurred first.
        if day_low <= stop_price and day_high >= target_price:
            exit_price = stop_price
            reason = "both_assume_stop"

        elif day_low <= stop_price:
            exit_price = stop_price
            reason = "stop"

        elif day_high >= target_price:
            exit_price = target_price
            reason = "target"

        else:
            exit_price = float(next_day["Close"])
            reason = "time"

        trades.append(
            {
                "symbol": symbol,
                "signal_date": str(df.index[i].date()),
                "entry_date": str(df.index[i + 1].date()),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "return_pct": (
                    exit_price / entry_price - 1
                )
                * 100,
                "reason": reason,
            }
        )

    return trades


def run_symbol(
    symbol: str,
    start: str,
    end: str | None,
    target: float,
) -> tuple[str, list[dict], str | None]:
    try:
        trades = test_symbol(
            symbol=symbol,
            start=start,
            end=end,
            target=target,
        )
        return symbol, trades, None

    except Exception as exc:
        return symbol, [], str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        default="2021-01-01",
    )

    parser.add_argument(
        "--end",
        default=None,
    )

    parser.add_argument(
        "--target",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
    )

    args = parser.parse_args()

    symbols = list(dict.fromkeys(nifty100_symbols()))

    print(f"Symbols: {len(symbols)}")
    print(f"Start: {args.start}")
    print(f"End: {args.end or 'latest'}")
    print(f"Target: {args.target}%")
    print(f"Parallel workers: {args.workers}")

    all_trades: list[dict] = []
    completed = 0
    failed = 0

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = {
            executor.submit(
                run_symbol,
                symbol,
                args.start,
                args.end,
                args.target,
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):
            symbol, symbol_trades, error = future.result()

            completed += 1

            if error:
                failed += 1
                print(
                    f"[{completed}/{len(symbols)}] "
                    f"{symbol}: ERROR — {error}"
                )
            else:
                all_trades.extend(symbol_trades)
                print(
                    f"[{completed}/{len(symbols)}] "
                    f"{symbol}: {len(symbol_trades)} trades"
                )

    output_file = (
        f"backtest_target_{args.target}.csv"
    )

    output = pd.DataFrame(all_trades)

    if output.empty:
        output.to_csv(output_file, index=False)

        print("\nBacktest completed.")
        print("No trades found.")
        print(f"Failed symbols: {failed}")
        print(f"CSV: {output_file}")
        return

    output["entry_date"] = pd.to_datetime(
        output["entry_date"]
    )

    output = output.sort_values(
        ["entry_date", "symbol"]
    ).reset_index(drop=True)

    output["entry_date"] = output[
        "entry_date"
    ].dt.strftime("%Y-%m-%d")

    output.to_csv(output_file, index=False)

    equity = (
        1 + output["return_pct"] / 100
    ).cumprod()

    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1

    wins = output["return_pct"] > 0
    losses = output["return_pct"] < 0

    print("\n========== BACKTEST RESULT ==========")
    print("Trades:", len(output))
    print(
        "Winning trades:",
        int(wins.sum()),
    )
    print(
        "Losing trades:",
        int(losses.sum()),
    )
    print(
        "Win rate:",
        round(wins.mean() * 100, 2),
        "%",
    )
    print(
        "Average/trade:",
        round(output["return_pct"].mean(), 3),
        "%",
    )
    print(
        "Compounded:",
        round((equity.iloc[-1] - 1) * 100, 2),
        "%",
    )
    print(
        "Max drawdown:",
        round(drawdown.min() * 100, 2),
        "%",
    )
    print(
        "Failed symbols:",
        failed,
    )

    print("\nExit reasons:")
    print(output["reason"].value_counts())

    print("\nCSV saved as:", output_file)
    print("=====================================")


if __name__ == "__main__":
    main()
