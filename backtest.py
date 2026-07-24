"""
Nifty 100 V4 conservative price-only backtest.

Important limitations:
- Uses the current Nifty 100 list, which creates survivorship bias.
- Does not use historical point-in-time fundamentals or news.
- Daily candles cannot exactly reproduce the live first-15-minute breakout.
- If both stop and target occur in the same daily candle, the stop is
  conservatively assumed to have occurred first.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import pandas as pd
import yfinance as yf

import config
from data_provider import nifty100_symbols
from indicators import rsi


BATCH_SIZE = 8
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
DOWNLOAD_TIMEOUT_SECONDS = 30


def clean_symbols(symbols: list[str]) -> list[str]:
    """Remove blanks and duplicate symbols while preserving order."""

    cleaned: list[str] = []

    for symbol in symbols:
        symbol = str(symbol).strip()

        if symbol and symbol not in cleaned:
            cleaned.append(symbol)

    return cleaned


def split_batches(items: list[str], size: int) -> list[list[str]]:
    """Split symbols into smaller download batches."""

    return [
        items[index : index + size]
        for index in range(0, len(items), size)
    ]


def normalize_single_ticker_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Convert one ticker's Yahoo data into standard OHLCV columns."""

    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    df = dataframe.copy()

    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels) > 1:
            try:
                df.columns = df.columns.get_level_values(-1)
            except Exception:
                pass

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    if not all(column in df.columns for column in required_columns):
        return pd.DataFrame()

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["Open", "High", "Low", "Close", "Volume"]
    )

    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    return df


def extract_symbol_dataframe(
    downloaded: pd.DataFrame,
    symbol: str,
    batch_size: int,
) -> pd.DataFrame:
    """Extract one ticker from a Yahoo multi-ticker response."""

    if downloaded is None or downloaded.empty:
        return pd.DataFrame()

    try:
        if batch_size == 1:
            return normalize_single_ticker_dataframe(downloaded)

        if not isinstance(downloaded.columns, pd.MultiIndex):
            return pd.DataFrame()

        # Expected because group_by="ticker" is used.
        if symbol in downloaded.columns.get_level_values(0):
            symbol_df = downloaded[symbol].copy()
            return normalize_single_ticker_dataframe(symbol_df)

        # Fallback for alternative yfinance column ordering.
        if symbol in downloaded.columns.get_level_values(-1):
            symbol_df = downloaded.xs(
                symbol,
                axis=1,
                level=-1,
                drop_level=True,
            )
            return normalize_single_ticker_dataframe(symbol_df)

    except Exception as exc:
        print(
            f"{symbol}: extraction error: {exc}",
            flush=True,
        )

    return pd.DataFrame()


def download_batch(
    symbols: list[str],
    start: str,
    end: str | None,
) -> dict[str, pd.DataFrame]:
    """Download one small batch with retry protection."""

    result: dict[str, pd.DataFrame] = {
        symbol: pd.DataFrame()
        for symbol in symbols
    }

    ticker_argument: str | list[str]

    if len(symbols) == 1:
        ticker_argument = symbols[0]
    else:
        ticker_argument = symbols

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"Downloading batch of {len(symbols)} symbols "
                f"(attempt {attempt}/{MAX_RETRIES})...",
                flush=True,
            )

            downloaded = yf.download(
                tickers=ticker_argument,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )

            successful_count = 0

            for symbol in symbols:
                symbol_df = extract_symbol_dataframe(
                    downloaded=downloaded,
                    symbol=symbol,
                    batch_size=len(symbols),
                )

                if not symbol_df.empty:
                    result[symbol] = symbol_df
                    successful_count += 1

            if successful_count == len(symbols):
                return result

            print(
                f"Batch returned {successful_count}/{len(symbols)} "
                "usable symbols.",
                flush=True,
            )

        except Exception as exc:
            print(
                f"Batch download failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

        if attempt < MAX_RETRIES:
            wait_seconds = RETRY_DELAY_SECONDS * attempt

            print(
                f"Waiting {wait_seconds} seconds before retry...",
                flush=True,
            )

            time.sleep(wait_seconds)

    return result


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators required by the V4 setup."""

    prepared = df.copy()

    prepared["change"] = prepared["Close"].pct_change() * 100
    prepared["dma200"] = prepared["Close"].rolling(200).mean()
    prepared["rsi"] = rsi(prepared["Close"])

    prepared["avg_value"] = (
        prepared["Close"] * prepared["Volume"]
    ).rolling(20).mean()

    prepared["average_volume"] = (
        prepared["Volume"].rolling(20).mean()
    )

    prepared["volume_ratio"] = (
        prepared["Volume"] / prepared["average_volume"]
    )

    return prepared


def is_valid_number(value: Any) -> bool:
    """Return True only for usable numeric values."""

    try:
        return not pd.isna(value)
    except Exception:
        return False


def backtest_symbol(
    symbol: str,
    raw_df: pd.DataFrame,
    target_pct: float,
) -> list[dict[str, Any]]:
    """Run the strategy rules for one stock."""

    if raw_df is None or len(raw_df) < 260:
        return []

    df = prepare_indicators(raw_df)

    trades: list[dict[str, Any]] = []

    for index in range(252, len(df) - 1):
        row = df.iloc[index]
        next_day = df.iloc[index + 1]

        values_to_check = [
            row["Close"],
            row["Low"],
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

        if not all(
            is_valid_number(value)
            for value in values_to_check
        ):
            continue

        close_price = float(row["Close"])
        signal_low = float(row["Low"])
        dma200 = float(row["dma200"])
        daily_change = float(row["change"])
        stock_rsi = float(row["rsi"])
        average_traded_value = float(row["avg_value"])
        volume_ratio = float(row["volume_ratio"])

        if dma200 <= 0:
            continue

        distance_from_dma200 = (
            close_price / dma200 - 1
        ) * 100

        previous_252_day_low = float(
            df["Low"]
            .iloc[index - 251 : index + 1]
            .min()
        )

        near_new_52_week_low = (
            signal_low
            <= previous_252_day_low * 1.002
        )

        eligible = (
            config.MIN_DROP_PCT
            <= daily_change
            <= config.MAX_DROP_PCT

            and close_price
            >= config.MIN_PRICE

            and average_traded_value
            >= config.MIN_AVG_VALUE_20D

            and config.MIN_DISTANCE_200DMA
            <= distance_from_dma200
            <= config.MAX_DISTANCE_200DMA

            and config.MIN_RSI
            <= stock_rsi
            <= config.MAX_RSI

            and volume_ratio
            >= config.MIN_VOLUME_RATIO

            and not near_new_52_week_low
        )

        if not eligible:
            continue

        next_open = float(next_day["Open"])
        next_high = float(next_day["High"])
        next_low = float(next_day["Low"])
        next_close = float(next_day["Close"])

        gap_pct = (
            next_open / close_price - 1
        ) * 100

        if not (
            config.MIN_GAP_PCT
            <= gap_pct
            <= config.MAX_GAP_PCT
        ):
            continue

        # Daily candle approximation of the bullish first-15-minute rule.
        if (
            config.REQUIRE_BULLISH_FIRST_15M
            and next_close <= next_open
        ):
            continue

        entry_price = next_open

        stop_price = entry_price * (
            1 - config.STOP_LOSS_PCT / 100
        )

        target_price = entry_price * (
            1 + target_pct / 100
        )

        # Daily bars do not show whether stop or target happened first.
        # Assume stop first to avoid overstating performance.
        if (
            next_low <= stop_price
            and next_high >= target_price
        ):
            exit_price = stop_price
            exit_reason = "both_assume_stop"

        elif next_low <= stop_price:
            exit_price = stop_price
            exit_reason = "stop"

        elif next_high >= target_price:
            exit_price = target_price
            exit_reason = "target"

        else:
            exit_price = next_close
            exit_reason = "time"

        return_pct = (
            exit_price / entry_price - 1
        ) * 100

        trades.append(
            {
                "symbol": symbol,
                "signal_date": str(
                    df.index[index].date()
                ),
                "entry_date": str(
                    df.index[index + 1].date()
                ),
                "signal_change_pct": round(
                    daily_change,
                    3,
                ),
                "rsi": round(stock_rsi, 2),
                "distance_200dma_pct": round(
                    distance_from_dma200,
                    3,
                ),
                "volume_ratio": round(
                    volume_ratio,
                    3,
                ),
                "gap_pct": round(gap_pct, 3),
                "entry_price": round(
                    entry_price,
                    2,
                ),
                "exit_price": round(
                    exit_price,
                    2,
                ),
                "return_pct": round(
                    return_pct,
                    4,
                ),
                "reason": exit_reason,
            }
        )

    return trades


def calculate_summary(
    trades_df: pd.DataFrame,
) -> None:
    """Print performance statistics."""

    if trades_df.empty:
        print("\nNo trades found.", flush=True)
        return

    trades_df["entry_date_sort"] = pd.to_datetime(
        trades_df["entry_date"],
        errors="coerce",
    )

    trades_df.sort_values(
        by=["entry_date_sort", "symbol"],
        inplace=True,
    )

    trades_df.drop(
        columns=["entry_date_sort"],
        inplace=True,
    )

    trades_df.reset_index(
        drop=True,
        inplace=True,
    )

    returns = pd.to_numeric(
        trades_df["return_pct"],
        errors="coerce",
    ).fillna(0.0)

    equity_curve = (
        1 + returns / 100
    ).cumprod()

    running_peak = equity_curve.cummax()

    drawdown = (
        equity_curve / running_peak - 1
    )

    winning_trades = int(
        (returns > 0).sum()
    )

    losing_trades = int(
        (returns < 0).sum()
    )

    flat_trades = int(
        (returns == 0).sum()
    )

    win_rate = (
        winning_trades / len(returns) * 100
    )

    compounded_return = (
        equity_curve.iloc[-1] - 1
    ) * 100

    max_drawdown = (
        drawdown.min() * 100
    )

    print(
        "\n========== BACKTEST RESULT ==========",
        flush=True,
    )

    print(
        f"Trades: {len(trades_df)}",
        flush=True,
    )

    print(
        f"Winning trades: {winning_trades}",
        flush=True,
    )

    print(
        f"Losing trades: {losing_trades}",
        flush=True,
    )

    print(
        f"Flat trades: {flat_trades}",
        flush=True,
    )

    print(
        f"Win rate: {win_rate:.2f}%",
        flush=True,
    )

    print(
        f"Average/trade: {returns.mean():.3f}%",
        flush=True,
    )

    print(
        f"Compounded: {compounded_return:.2f}%",
        flush=True,
    )

    print(
        f"Max drawdown: {max_drawdown:.2f}%",
        flush=True,
    )

    print("\nExit reasons:", flush=True)

    print(
        trades_df["reason"].value_counts(),
        flush=True,
    )

    print(
        "=====================================",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        default="2023-01-01",
        help="Backtest start date: YYYY-MM-DD",
    )

    parser.add_argument(
        "--end",
        default=None,
        help="Optional end date: YYYY-MM-DD",
    )

    parser.add_argument(
        "--target",
        type=float,
        default=2.0,
        help="Target percentage",
    )

    args = parser.parse_args()

    symbols = clean_symbols(
        list(nifty100_symbols())
    )

    print(
        f"Total symbols: {len(symbols)}",
        flush=True,
    )

    print(
        f"Period: {args.start} to "
        f"{args.end or 'latest'}",
        flush=True,
    )

    print(
        f"Target: {args.target}%",
        flush=True,
    )

    batches = split_batches(
        symbols,
        BATCH_SIZE,
    )

    all_trades: list[dict[str, Any]] = []
    failed_symbols: list[str] = []

    processed_symbols = 0

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):
        print(
            f"\nBatch {batch_number}/{len(batches)}: "
            f"{', '.join(batch)}",
            flush=True,
        )

        batch_data = download_batch(
            symbols=batch,
            start=args.start,
            end=args.end,
        )

        for symbol in batch:
            processed_symbols += 1
            symbol_df = batch_data.get(
                symbol,
                pd.DataFrame(),
            )

            if symbol_df.empty:
                failed_symbols.append(symbol)

                print(
                    f"[{processed_symbols}/{len(symbols)}] "
                    f"{symbol}: no usable data",
                    flush=True,
                )

                continue

            try:
                symbol_trades = backtest_symbol(
                    symbol=symbol,
                    raw_df=symbol_df,
                    target_pct=args.target,
                )

                all_trades.extend(symbol_trades)

                print(
                    f"[{processed_symbols}/{len(symbols)}] "
                    f"{symbol}: "
                    f"{len(symbol_trades)} trades",
                    flush=True,
                )

            except Exception as exc:
                failed_symbols.append(symbol)

                print(
                    f"[{processed_symbols}/{len(symbols)}] "
                    f"{symbol}: ERROR "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        # Small pause reduces Yahoo Finance rate limiting.
        time.sleep(2)

    output_filename = (
        f"backtest_target_{args.target}.csv"
    )

    output_df = pd.DataFrame(all_trades)

    if not output_df.empty:
        output_df["entry_date_sort"] = pd.to_datetime(
            output_df["entry_date"],
            errors="coerce",
        )

        output_df.sort_values(
            by=["entry_date_sort", "symbol"],
            inplace=True,
        )

        output_df.drop(
            columns=["entry_date_sort"],
            inplace=True,
        )

        output_df.reset_index(
            drop=True,
            inplace=True,
        )

    output_df.to_csv(
        output_filename,
        index=False,
    )

    calculate_summary(output_df)

    print(
        f"\nCSV saved: {output_filename}",
        flush=True,
    )

    print(
        f"Successful symbols: "
        f"{len(symbols) - len(failed_symbols)}",
        flush=True,
    )

    print(
        f"Failed symbols: {len(failed_symbols)}",
        flush=True,
    )

    if failed_symbols:
        print(
            "Failed symbol list: "
            + ", ".join(failed_symbols),
            flush=True,
        )

    print(
        "\nBacktest completed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
