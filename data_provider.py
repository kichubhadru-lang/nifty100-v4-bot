from __future__ import annotations
from typing import Any
import pandas as pd
import yfinance as yf
import config
from utils import flatten_yf_columns


def nifty100_symbols() -> list[str]:
    df = pd.read_csv(config.NIFTY100_CSV)
    symbol_column = next(
        (c for c in df.columns if c.strip().lower() == "symbol"), None
    )
    if symbol_column is None:
        raise RuntimeError(f"Nifty 100 CSV has no Symbol column: {list(df.columns)}")
    return [f"{s.strip()}.NS" for s in df[symbol_column].dropna().astype(str)]


def daily(symbol: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    return flatten_yf_columns(df).dropna()


def intraday(symbol: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
        prepost=False,
    )
    return flatten_yf_columns(df).dropna()


def ticker_info(symbol: str) -> dict[str, Any]:
    return yf.Ticker(symbol).get_info()


def recent_news(symbol: str) -> list[dict[str, Any]]:
    try:
        return yf.Ticker(symbol).get_news(count=10) or []
    except Exception:
        return []
