"""
Nifty 100 V4 realistic portfolio backtest.

Features:
- Robust batched Yahoo Finance downloads
- One trade per day
- No overlapping positions
- Risk-based position sizing
- Fixed starting capital
- Trading costs and slippage
- Equity curve
- Monthly return report
- Profit factor
- Expectancy
- Maximum consecutive losses
- Sharpe ratio
- Conservative daily-candle execution assumptions

Limitations:
- Uses current Nifty 100 constituents, creating survivorship bias.
- Does not use point-in-time fundamentals.
- Daily candles cannot exactly reproduce the first-15-minute breakout.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

import config
from data_provider import nifty100_symbols
from indicators import rsi


# ============================================================
# DEFAULT PORTFOLIO SETTINGS
# ============================================================

DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_RISK_PER_TRADE_PCT = 1.0
DEFAULT_MAX_ALLOCATION_PCT = 100.0
DEFAULT_ROUND_TRIP_COST_PCT = 0.30

BATCH_SIZE = 8
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 4
DOWNLOAD_TIMEOUT_SECONDS = 30

OUTPUT_DIRECTORY = Path("data/backtest")


# ============================================================
# SYMBOL AND DOWNLOAD HELPERS
# ============================================================

def clean_symbols(symbols: list[str]) -> list[str]:
    """Remove blank and duplicate symbols."""

    cleaned: list[str] = []

    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()

        if not symbol:
            continue

        if not symbol.endswith(".NS"):
            symbol = f"{symbol}.NS"

        if symbol not in cleaned:
            cleaned.append(symbol)

    return cleaned


def split_batches(
    items: list[str],
    batch_size: int,
) -> list[list[str]]:
    """Split a list into smaller batches."""

    return [
        items[index:index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def normalize_price_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Convert downloaded data to standard OHLCV columns."""

    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    df = dataframe.copy()

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(-1)
        except Exception:
            return pd.DataFrame()

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):
        return pd.DataFrame()

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required_columns,
    )

    df = df[
        ~df.index.duplicated(keep="last")
    ]

    df = df.sort_index()

    try:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    except TypeError:
        df.index = pd.to_datetime(df.index)

    return df


def extract_symbol_dataframe(
    downloaded: pd.DataFrame,
    symbol: str,
    number_of_symbols: int,
) -> pd.DataFrame:
    """Extract one stock from a Yahoo multi-symbol response."""

    if downloaded is None or downloaded.empty:
        return pd.DataFrame()

    try:
        if number_of_symbols == 1:
            return normalize_price_dataframe(downloaded)

        if not isinstance(downloaded.columns, pd.MultiIndex):
            return pd.DataFrame()

        first_level = downloaded.columns.get_level_values(0)
        last_level = downloaded.columns.get_level_values(-1)

        if symbol in first_level:
            symbol_df = downloaded[symbol].copy()
            return normalize_price_dataframe(symbol_df)

        if symbol in last_level:
            symbol_df = downloaded.xs(
                symbol,
                axis=1,
                level=-1,
                drop_level=True,
            )

            return normalize_price_dataframe(symbol_df)

    except Exception as exc:
        print(
            f"{symbol}: extraction failed: {exc}",
            flush=True,
        )

    return pd.DataFrame()


def download_batch(
    symbols: list[str],
    start_date: str,
    end_date: str | None,
) -> dict[str, pd.DataFrame]:
    """Download a small group of symbols with retries."""

    result = {
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
                f"Downloading {len(symbols)} symbols "
                f"(attempt {attempt}/{MAX_RETRIES})",
                flush=True,
            )

            downloaded = yf.download(
                tickers=ticker_argument,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )

            successful = 0

            for symbol in symbols:
                symbol_df = extract_symbol_dataframe(
                    downloaded=downloaded,
                    symbol=symbol,
                    number_of_symbols=len(symbols),
                )

                if not symbol_df.empty:
                    result[symbol] = symbol_df
                    successful += 1

            if successful == len(symbols):
                return result

            print(
                f"Received usable data for "
                f"{successful}/{len(symbols)} symbols.",
                flush=True,
            )

        except Exception as exc:
            print(
                f"Batch download error: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

        if attempt < MAX_RETRIES:
            wait_seconds = RETRY_DELAY_SECONDS * attempt

            print(
                f"Retrying after {wait_seconds} seconds...",
                flush=True,
            )

            time.sleep(wait_seconds)

    return result


def download_all_symbols(
    symbols: list[str],
    start_date: str,
    end_date: str | None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Download all requested symbols in controlled batches."""

    all_data: dict[str, pd.DataFrame] = {}
    failed_symbols: list[str] = []

    batches = split_batches(
        symbols,
        BATCH_SIZE,
    )

    processed = 0

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):
        print(
            f"\nBatch {batch_number}/{len(batches)}",
            flush=True,
        )

        batch_result = download_batch(
            symbols=batch,
            start_date=start_date,
            end_date=end_date,
        )

        for symbol in batch:
            processed += 1

            symbol_df = batch_result.get(
                symbol,
                pd.DataFrame(),
            )

            if symbol_df.empty:
                failed_symbols.append(symbol)

                print(
                    f"[{processed}/{len(symbols)}] "
                    f"{symbol}: no data",
                    flush=True,
                )
            else:
                all_data[symbol] = symbol_df

                print(
                    f"[{processed}/{len(symbols)}] "
                    f"{symbol}: {len(symbol_df)} candles",
                    flush=True,
                )

        time.sleep(1)

    return all_data, failed_symbols


# ============================================================
# INDICATORS AND SIGNAL GENERATION
# ============================================================

def prepare_indicators(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate strategy indicators."""

    df = raw_df.copy()

    df["daily_change_pct"] = (
        df["Close"].pct_change() * 100
    )

    df["dma200"] = (
        df["Close"].rolling(200).mean()
    )

    df["rsi"] = rsi(df["Close"])

    df["average_volume_20d"] = (
        df["Volume"].rolling(20).mean()
    )

    df["volume_ratio"] = (
        df["Volume"]
        / df["average_volume_20d"]
    )

    df["average_traded_value_20d"] = (
        df["Close"] * df["Volume"]
    ).rolling(20).mean()

    df["low_252"] = (
        df["Low"].rolling(252).min()
    )

    return df


def calculate_setup_score(
    daily_change_pct: float,
    stock_rsi: float,
    distance_200dma_pct: float,
    volume_ratio: float,
) -> float:
    """
    Score qualifying candidates.

    Higher scores favour:
    - Larger controlled declines
    - Lower RSI
    - Strong volume
    - Price staying close to or above the 200 DMA
    """

    drop_score = min(
        max(abs(daily_change_pct), 0.0),
        10.0,
    ) * 3.0

    rsi_score = max(
        0.0,
        55.0 - stock_rsi,
    ) * 1.5

    volume_score = min(
        max(volume_ratio - 1.0, 0.0),
        3.0,
    ) * 15.0

    dma_score = max(
        0.0,
        10.0 - abs(distance_200dma_pct),
    )

    total_score = (
        drop_score
        + rsi_score
        + volume_score
        + dma_score
    )

    return round(total_score, 3)


def generate_candidates_for_symbol(
    symbol: str,
    raw_df: pd.DataFrame,
    target_pct: float,
) -> list[dict[str, Any]]:
    """Generate all historical candidates for one symbol."""

    if raw_df is None or len(raw_df) < 260:
        return []

    df = prepare_indicators(raw_df)

    candidates: list[dict[str, Any]] = []

    for index in range(252, len(df) - 1):
        signal_row = df.iloc[index]
        entry_row = df.iloc[index + 1]

        required_values = [
            signal_row["Close"],
            signal_row["Low"],
            signal_row["daily_change_pct"],
            signal_row["dma200"],
            signal_row["rsi"],
            signal_row["volume_ratio"],
            signal_row["average_traded_value_20d"],
            signal_row["low_252"],
            entry_row["Open"],
            entry_row["High"],
            entry_row["Low"],
            entry_row["Close"],
        ]

        if any(
            pd.isna(value)
            for value in required_values
        ):
            continue

        signal_close = float(signal_row["Close"])
        signal_low = float(signal_row["Low"])
        daily_change_pct = float(
            signal_row["daily_change_pct"]
        )

        dma200 = float(signal_row["dma200"])
        stock_rsi = float(signal_row["rsi"])
        volume_ratio = float(
            signal_row["volume_ratio"]
        )

        average_traded_value = float(
            signal_row["average_traded_value_20d"]
        )

        low_252 = float(signal_row["low_252"])

        if dma200 <= 0:
            continue

        distance_200dma_pct = (
            signal_close / dma200 - 1
        ) * 100

        near_52_week_low = (
            signal_low <= low_252 * 1.002
        )

        eligible = (
            config.MIN_DROP_PCT
            <= daily_change_pct
            <= config.MAX_DROP_PCT

            and signal_close
            >= config.MIN_PRICE

            and average_traded_value
            >= config.MIN_AVG_VALUE_20D

            and config.MIN_DISTANCE_200DMA
            <= distance_200dma_pct
            <= config.MAX_DISTANCE_200DMA

            and config.MIN_RSI
            <= stock_rsi
            <= config.MAX_RSI

            and volume_ratio
            >= config.MIN_VOLUME_RATIO

            and not near_52_week_low
        )

        if not eligible:
            continue

        entry_open = float(entry_row["Open"])
        entry_high = float(entry_row["High"])
        entry_low = float(entry_row["Low"])
        entry_close = float(entry_row["Close"])

        gap_pct = (
            entry_open / signal_close - 1
        ) * 100

        if not (
            config.MIN_GAP_PCT
            <= gap_pct
            <= config.MAX_GAP_PCT
        ):
            continue

        # Daily-candle approximation of the live confirmation rule.
        if (
            config.REQUIRE_BULLISH_FIRST_15M
            and entry_close <= entry_open
        ):
            continue

        setup_score = calculate_setup_score(
            daily_change_pct=daily_change_pct,
            stock_rsi=stock_rsi,
            distance_200dma_pct=distance_200dma_pct,
            volume_ratio=volume_ratio,
        )

        candidates.append(
            {
                "symbol": symbol,
                "signal_date": pd.Timestamp(
                    df.index[index]
                ),
                "entry_date": pd.Timestamp(
                    df.index[index + 1]
                ),
                "signal_close": signal_close,
                "signal_change_pct": daily_change_pct,
                "rsi": stock_rsi,
                "distance_200dma_pct": distance_200dma_pct,
                "volume_ratio": volume_ratio,
                "gap_pct": gap_pct,
                "setup_score": setup_score,
                "entry_open": entry_open,
                "entry_high": entry_high,
                "entry_low": entry_low,
                "entry_close": entry_close,
                "target_pct": target_pct,
            }
        )

    return candidates


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def determine_gross_exit(
    candidate: pd.Series,
    target_pct: float,
    stop_loss_pct: float,
) -> tuple[float, str]:
    """Determine the gross exit price using daily OHLC data."""

    entry_price = float(candidate["entry_open"])
    day_high = float(candidate["entry_high"])
    day_low = float(candidate["entry_low"])
    day_close = float(candidate["entry_close"])

    stop_price = entry_price * (
        1 - stop_loss_pct / 100
    )

    target_price = entry_price * (
        1 + target_pct / 100
    )

    if (
        day_low <= stop_price
        and day_high >= target_price
    ):
        return stop_price, "both_assume_stop"

    if day_low <= stop_price:
        return stop_price, "stop"

    if day_high >= target_price:
        return target_price, "target"

    return day_close, "time"


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss_pct: float,
    risk_per_trade_pct: float,
    max_allocation_pct: float,
) -> tuple[int, float, float]:
    """
    Calculate position size using both:
    - risk limit
    - capital allocation limit
    """

    if (
        capital <= 0
        or entry_price <= 0
        or stop_loss_pct <= 0
    ):
        return 0, 0.0, 0.0

    risk_budget = capital * (
        risk_per_trade_pct / 100
    )

    risk_per_share = entry_price * (
        stop_loss_pct / 100
    )

    shares_by_risk = math.floor(
        risk_budget / risk_per_share
    )

    maximum_investment = capital * (
        max_allocation_pct / 100
    )

    shares_by_capital = math.floor(
        maximum_investment / entry_price
    )

    quantity = min(
        shares_by_risk,
        shares_by_capital,
    )

    if quantity <= 0:
        return 0, risk_budget, 0.0

    invested_amount = quantity * entry_price

    return quantity, risk_budget, invested_amount


def simulate_portfolio(
    candidates_df: pd.DataFrame,
    initial_capital: float,
    target_pct: float,
    stop_loss_pct: float,
    risk_per_trade_pct: float,
    max_allocation_pct: float,
    round_trip_cost_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate one selected trade per day.

    The candidate with the highest setup score is selected.
    Since this strategy exits on the entry day, there are no
    overnight overlapping positions.
    """

    if candidates_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    candidates = candidates_df.copy()

    candidates["entry_date"] = pd.to_datetime(
        candidates["entry_date"]
    )

    candidates = candidates.sort_values(
        by=[
            "entry_date",
            "setup_score",
            "volume_ratio",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    # Keep only the best candidate for each trading day.
    selected = (
        candidates
        .groupby("entry_date", as_index=False)
        .first()
        .sort_values("entry_date")
        .reset_index(drop=True)
    )

    capital = float(initial_capital)

    trade_records: list[dict[str, Any]] = []
    equity_records: list[dict[str, Any]] = []

    for _, candidate in selected.iterrows():
        entry_price = float(candidate["entry_open"])

        quantity, risk_budget, invested_amount = (
            calculate_position_size(
                capital=capital,
                entry_price=entry_price,
                stop_loss_pct=stop_loss_pct,
                risk_per_trade_pct=risk_per_trade_pct,
                max_allocation_pct=max_allocation_pct,
            )
        )

        if quantity <= 0:
            continue

        gross_exit_price, exit_reason = determine_gross_exit(
            candidate=candidate,
            target_pct=target_pct,
            stop_loss_pct=stop_loss_pct,
        )

        gross_pnl = (
            gross_exit_price - entry_price
        ) * quantity

        estimated_cost = invested_amount * (
            round_trip_cost_pct / 100
        )

        net_pnl = gross_pnl - estimated_cost

        capital_before = capital
        capital += net_pnl

        gross_return_pct = (
            gross_exit_price / entry_price - 1
        ) * 100

        net_return_on_position_pct = (
            net_pnl / invested_amount * 100
            if invested_amount > 0
            else 0.0
        )

        portfolio_return_pct = (
            net_pnl / capital_before * 100
            if capital_before > 0
            else 0.0
        )

        trade_record = {
            "symbol": candidate["symbol"],
            "signal_date": pd.Timestamp(
                candidate["signal_date"]
            ).strftime("%Y-%m-%d"),
            "entry_date": pd.Timestamp(
                candidate["entry_date"]
            ).strftime("%Y-%m-%d"),
            "setup_score": round(
                float(candidate["setup_score"]),
                3,
            ),
            "signal_change_pct": round(
                float(candidate["signal_change_pct"]),
                3,
            ),
            "rsi": round(
                float(candidate["rsi"]),
                2,
            ),
            "distance_200dma_pct": round(
                float(candidate["distance_200dma_pct"]),
                3,
            ),
            "volume_ratio": round(
                float(candidate["volume_ratio"]),
                3,
            ),
            "gap_pct": round(
                float(candidate["gap_pct"]),
                3,
            ),
            "quantity": quantity,
            "entry_price": round(entry_price, 2),
            "gross_exit_price": round(
                gross_exit_price,
                2,
            ),
            "invested_amount": round(
                invested_amount,
                2,
            ),
            "risk_budget": round(
                risk_budget,
                2,
            ),
            "estimated_cost": round(
                estimated_cost,
                2,
            ),
            "gross_return_pct": round(
                gross_return_pct,
                4,
            ),
            "net_position_return_pct": round(
                net_return_on_position_pct,
                4,
            ),
            "portfolio_return_pct": round(
                portfolio_return_pct,
                4,
            ),
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "capital_before": round(
                capital_before,
                2,
            ),
            "capital_after": round(
                capital,
                2,
            ),
            "exit_reason": exit_reason,
        }

        trade_records.append(trade_record)

        equity_records.append(
            {
                "date": trade_record["entry_date"],
                "capital": round(capital, 2),
                "net_pnl": round(net_pnl, 2),
                "portfolio_return_pct": round(
                    portfolio_return_pct,
                    4,
                ),
            }
        )

        if capital <= 0:
            print(
                "Portfolio capital depleted. "
                "Stopping simulation.",
                flush=True,
            )
            break

    return (
        pd.DataFrame(trade_records),
        pd.DataFrame(equity_records),
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def maximum_consecutive_losses(
    pnl_series: pd.Series,
) -> int:
    """Calculate the longest losing streak."""

    longest_streak = 0
    current_streak = 0

    for pnl in pnl_series:
        if pnl < 0:
            current_streak += 1
            longest_streak = max(
                longest_streak,
                current_streak,
            )
        else:
            current_streak = 0

    return longest_streak


def calculate_monthly_returns(
    equity_df: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    """Calculate month-end portfolio returns."""

    if equity_df.empty:
        return pd.DataFrame()

    monthly = equity_df.copy()

    monthly["date"] = pd.to_datetime(
        monthly["date"]
    )

    monthly = monthly.set_index("date")

    month_end_capital = (
        monthly["capital"]
        .resample("ME")
        .last()
        .dropna()
    )

    previous_capital = (
        month_end_capital
        .shift(1)
        .fillna(initial_capital)
    )

    monthly_return_pct = (
        month_end_capital / previous_capital - 1
    ) * 100

    result = pd.DataFrame(
        {
            "month": month_end_capital.index.strftime(
                "%Y-%m"
            ),
            "ending_capital": month_end_capital.values,
            "monthly_return_pct": monthly_return_pct.values,
        }
    )

    result["ending_capital"] = result[
        "ending_capital"
    ].round(2)

    result["monthly_return_pct"] = result[
        "monthly_return_pct"
    ].round(3)

    return result


def calculate_metrics(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float,
) -> dict[str, Any]:
    """Calculate professional portfolio metrics."""

    if trades_df.empty:
        return {
            "trades": 0,
            "final_capital": initial_capital,
        }

    net_pnl = pd.to_numeric(
        trades_df["net_pnl"],
        errors="coerce",
    ).fillna(0.0)

    portfolio_returns = pd.to_numeric(
        trades_df["portfolio_return_pct"],
        errors="coerce",
    ).fillna(0.0) / 100

    winners = net_pnl[net_pnl > 0]
    losers = net_pnl[net_pnl < 0]

    winning_trades = int((net_pnl > 0).sum())
    losing_trades = int((net_pnl < 0).sum())
    flat_trades = int((net_pnl == 0).sum())

    win_rate = (
        winning_trades / len(net_pnl) * 100
    )

    gross_profit = float(winners.sum())
    gross_loss = abs(float(losers.sum()))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf")

    average_win = (
        float(winners.mean())
        if not winners.empty
        else 0.0
    )

    average_loss = (
        float(losers.mean())
        if not losers.empty
        else 0.0
    )

    expectancy = float(net_pnl.mean())

    final_capital = float(
        trades_df["capital_after"].iloc[-1]
    )

    total_return_pct = (
        final_capital / initial_capital - 1
    ) * 100

    equity_series = pd.concat(
        [
            pd.Series([initial_capital]),
            pd.to_numeric(
                equity_df["capital"],
                errors="coerce",
            ),
        ],
        ignore_index=True,
    )

    rolling_peak = equity_series.cummax()

    drawdown = (
        equity_series / rolling_peak - 1
    )

    max_drawdown_pct = (
        float(drawdown.min()) * 100
    )

    return_std = float(
        portfolio_returns.std(ddof=1)
    )

    if (
        len(portfolio_returns) > 1
        and return_std > 0
    ):
        sharpe_ratio = (
            portfolio_returns.mean()
            / return_std
            * math.sqrt(252)
        )
    else:
        sharpe_ratio = 0.0

    max_consecutive_losses = (
        maximum_consecutive_losses(net_pnl)
    )

    return {
        "trades": len(trades_df),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "flat_trades": flat_trades,
        "win_rate_pct": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "average_win": average_win,
        "average_loss": average_loss,
        "expectancy_per_trade": expectancy,
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "net_profit": final_capital - initial_capital,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "max_consecutive_losses": max_consecutive_losses,
        "total_estimated_cost": float(
            trades_df["estimated_cost"].sum()
        ),
    }


def print_metrics(
    metrics: dict[str, Any],
    target_pct: float,
    stop_loss_pct: float,
    risk_per_trade_pct: float,
    round_trip_cost_pct: float,
) -> None:
    """Print the final performance summary."""

    print(
        "\n========== REALISTIC BACKTEST ==========",
        flush=True,
    )

    print(
        f"Target: {target_pct:.2f}%",
        flush=True,
    )

    print(
        f"Stop loss: {stop_loss_pct:.2f}%",
        flush=True,
    )

    print(
        f"Risk per trade: {risk_per_trade_pct:.2f}%",
        flush=True,
    )

    print(
        f"Estimated round-trip cost: "
        f"{round_trip_cost_pct:.2f}%",
        flush=True,
    )

    print(
        f"Trades: {metrics.get('trades', 0)}",
        flush=True,
    )

    if metrics.get("trades", 0) == 0:
        print(
            "No portfolio trades found.",
            flush=True,
        )
        return

    print(
        f"Winning trades: "
        f"{metrics['winning_trades']}",
        flush=True,
    )

    print(
        f"Losing trades: "
        f"{metrics['losing_trades']}",
        flush=True,
    )

    print(
        f"Win rate: "
        f"{metrics['win_rate_pct']:.2f}%",
        flush=True,
    )

    print(
        f"Profit factor: "
        f"{metrics['profit_factor']:.2f}",
        flush=True,
    )

    print(
        f"Average win: "
        f"₹{metrics['average_win']:.2f}",
        flush=True,
    )

    print(
        f"Average loss: "
        f"₹{metrics['average_loss']:.2f}",
        flush=True,
    )

    print(
        f"Expectancy/trade: "
        f"₹{metrics['expectancy_per_trade']:.2f}",
        flush=True,
    )

    print(
        f"Initial capital: "
        f"₹{metrics['initial_capital']:.2f}",
        flush=True,
    )

    print(
        f"Final capital: "
        f"₹{metrics['final_capital']:.2f}",
        flush=True,
    )

    print(
        f"Net profit: "
        f"₹{metrics['net_profit']:.2f}",
        flush=True,
    )

    print(
        f"Total return: "
        f"{metrics['total_return_pct']:.2f}%",
        flush=True,
    )

    print(
        f"Maximum drawdown: "
        f"{metrics['max_drawdown_pct']:.2f}%",
        flush=True,
    )

    print(
        f"Sharpe ratio: "
        f"{metrics['sharpe_ratio']:.2f}",
        flush=True,
    )

    print(
        f"Maximum consecutive losses: "
        f"{metrics['max_consecutive_losses']}",
        flush=True,
    )

    print(
        f"Estimated total costs: "
        f"₹{metrics['total_estimated_cost']:.2f}",
        flush=True,
    )

    print(
        "========================================",
        flush=True,
    )


def save_summary(
    metrics: dict[str, Any],
    filepath: Path,
) -> None:
    """Save metrics as a text report."""

    lines = [
        "NIFTY 100 V4 REALISTIC BACKTEST",
        "",
    ]

    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(
                f"{key}: {value:.4f}"
            )
        else:
            lines.append(
                f"{key}: {value}"
            )

    filepath.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        default="2023-01-01",
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
        "--capital",
        type=float,
        default=DEFAULT_INITIAL_CAPITAL,
    )

    parser.add_argument(
        "--risk",
        type=float,
        default=DEFAULT_RISK_PER_TRADE_PCT,
    )

    parser.add_argument(
        "--max-allocation",
        type=float,
        default=DEFAULT_MAX_ALLOCATION_PCT,
    )

    parser.add_argument(
        "--cost",
        type=float,
        default=DEFAULT_ROUND_TRIP_COST_PCT,
    )

    args = parser.parse_args()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    symbols = clean_symbols(
        list(nifty100_symbols())
    )

    print(
        f"Symbols: {len(symbols)}",
        flush=True,
    )

    print(
        f"Period: {args.start} to "
        f"{args.end or 'latest'}",
        flush=True,
    )

    all_price_data, failed_symbols = (
        download_all_symbols(
            symbols=symbols,
            start_date=args.start,
            end_date=args.end,
        )
    )

    all_candidates: list[dict[str, Any]] = []

    processed = 0

    for symbol, symbol_df in all_price_data.items():
        processed += 1

        try:
            candidates = generate_candidates_for_symbol(
                symbol=symbol,
                raw_df=symbol_df,
                target_pct=args.target,
            )

            all_candidates.extend(candidates)

            print(
                f"Signal scan "
                f"[{processed}/{len(all_price_data)}] "
                f"{symbol}: {len(candidates)} candidates",
                flush=True,
            )

        except Exception as exc:
            failed_symbols.append(symbol)

            print(
                f"{symbol}: signal generation failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    candidates_df = pd.DataFrame(
        all_candidates
    )

    candidates_file = (
        OUTPUT_DIRECTORY
        / "backtest_all_candidates.csv"
    )

    candidates_df.to_csv(
        candidates_file,
        index=False,
    )

    trades_df, equity_df = simulate_portfolio(
        candidates_df=candidates_df,
        initial_capital=args.capital,
        target_pct=args.target,
        stop_loss_pct=float(
            config.STOP_LOSS_PCT
        ),
        risk_per_trade_pct=args.risk,
        max_allocation_pct=args.max_allocation,
        round_trip_cost_pct=args.cost,
    )

    trades_file = (
        OUTPUT_DIRECTORY
        / "backtest_realistic_trades.csv"
    )

    equity_file = (
        OUTPUT_DIRECTORY
        / "backtest_equity_curve.csv"
    )

    monthly_file = (
        OUTPUT_DIRECTORY
        / "backtest_monthly_returns.csv"
    )

    summary_file = (
        OUTPUT_DIRECTORY
        / "backtest_summary.txt"
    )

    trades_df.to_csv(
        trades_file,
        index=False,
    )

    equity_df.to_csv(
        equity_file,
        index=False,
    )

    monthly_df = calculate_monthly_returns(
        equity_df=equity_df,
        initial_capital=args.capital,
    )

    monthly_df.to_csv(
        monthly_file,
        index=False,
    )

    metrics = calculate_metrics(
        trades_df=trades_df,
        equity_df=equity_df,
        initial_capital=args.capital,
    )

    print_metrics(
        metrics=metrics,
        target_pct=args.target,
        stop_loss_pct=float(
            config.STOP_LOSS_PCT
        ),
        risk_per_trade_pct=args.risk,
        round_trip_cost_pct=args.cost,
    )

    save_summary(
        metrics=metrics,
        filepath=summary_file,
    )

    print(
        f"\nRaw candidates: {len(candidates_df)}",
        flush=True,
    )

    print(
        f"Selected portfolio trades: {len(trades_df)}",
        flush=True,
    )

    print(
        f"Successful symbols: {len(all_price_data)}",
        flush=True,
    )

    print(
        f"Failed symbols: {len(set(failed_symbols))}",
        flush=True,
    )

    print("\nFiles created:", flush=True)

    print(
        f"- {candidates_file}",
        flush=True,
    )

    print(
        f"- {trades_file}",
        flush=True,
    )

    print(
        f"- {equity_file}",
        flush=True,
    )

    print(
        f"- {monthly_file}",
        flush=True,
    )

    print(
        f"- {summary_file}",
        flush=True,
    )

    print(
        "\nProfessional backtest completed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
