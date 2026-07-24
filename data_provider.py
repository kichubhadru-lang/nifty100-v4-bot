from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any
import time

import pandas as pd
import requests
import yfinance as yf

import config
from utils import flatten_yf_columns


CACHE_FILE = Path("data/nifty100_constituents.csv")

DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 20


# Used only when the official Nifty Indices CSV and local cache
# are both unavailable.
FALLBACK_NIFTY100 = [
    "ABB",
    "ADANIENSOL",
    "ADANIENT",
    "ADANIGREEN",
    "ADANIPORTS",
    "ADANIPOWER",
    "AMBUJACEM",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJAJFINSV",
    "BAJFINANCE",
    "BANKBARODA",
    "BEL",
    "BHARTIARTL",
    "BHEL",
    "BPCL",
    "BRITANNIA",
    "CANBK",
    "CGPOWER",
    "CHOLAFIN",
    "CIPLA",
    "COALINDIA",
    "DABUR",
    "DIVISLAB",
    "DLF",
    "DMART",
    "DRREDDY",
    "EICHERMOT",
    "GAIL",
    "GODREJCP",
    "GRASIM",
    "HAL",
    "HAVELLS",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "ICICIGI",
    "ICICIPRULI",
    "INDHOTEL",
    "INDIGO",
    "INDUSINDBK",
    "INFY",
    "IOC",
    "IRCTC",
    "IRFC",
    "ITC",
    "JINDALSTEL",
    "JIOFIN",
    "JSWENERGY",
    "JSWSTEEL",
    "KOTAKBANK",
    "LICHSGFIN",
    "LICI",
    "LODHA",
    "LT",
    "LTIM",
    "LTTS",
    "M&M",
    "MARICO",
    "MARUTI",
    "MAXHEALTH",
    "MOTHERSON",
    "NAUKRI",
    "NESTLEIND",
    "NHPC",
    "NTPC",
    "ONGC",
    "PFC",
    "PIDILITIND",
    "PNB",
    "POLYCAB",
    "POWERGRID",
    "RECLTD",
    "RELIANCE",
    "SBICARD",
    "SBILIFE",
    "SBIN",
    "SHREECEM",
    "SHRIRAMFIN",
    "SIEMENS",
    "SOLARINDS",
    "SUNPHARMA",
    "TATACONSUM",
    "TATAMOTORS",
    "TATAPOWER",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "TORNTPHARM",
    "TRENT",
    "TVSMOTOR",
    "ULTRACEMCO",
    "UNIONBANK",
    "UNITDSPR",
    "VBL",
    "VEDL",
    "WIPRO",
    "ZOMATO",
    "ZYDUSLIFE",
]


def normalize_symbols(symbols: list[str]) -> list[str]:
    """Clean symbols and add Yahoo Finance's .NS suffix."""

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


def parse_constituent_csv(text: str) -> list[str]:
    """Extract symbols from the official Nifty 100 CSV."""

    dataframe = pd.read_csv(StringIO(text))

    symbol_column = next(
        (
            column
            for column in dataframe.columns
            if column.strip().lower() == "symbol"
        ),
        None,
    )

    if symbol_column is None:
        raise RuntimeError(
            "Downloaded Nifty 100 CSV has no Symbol column. "
            f"Columns: {list(dataframe.columns)}"
        )

    symbols = (
        dataframe[symbol_column]
        .dropna()
        .astype(str)
        .tolist()
    )

    symbols = normalize_symbols(symbols)

    if len(symbols) < 50:
        raise RuntimeError(
            f"Only {len(symbols)} symbols were found in the CSV."
        )

    return symbols


def save_symbol_cache(symbols: list[str]) -> None:
    """Save the successful constituent list locally."""

    try:
        CACHE_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cache_dataframe = pd.DataFrame(
            {
                "Symbol": [
                    symbol.removesuffix(".NS")
                    for symbol in symbols
                ]
            }
        )

        cache_dataframe.to_csv(
            CACHE_FILE,
            index=False,
        )

        print(
            f"Saved Nifty 100 cache: {CACHE_FILE}",
            flush=True,
        )

    except Exception as exc:
        print(
            f"Could not save symbol cache: {exc}",
            flush=True,
        )


def load_symbol_cache() -> list[str]:
    """Read a previously saved constituent list."""

    if not CACHE_FILE.exists():
        return []

    try:
        dataframe = pd.read_csv(CACHE_FILE)

        symbol_column = next(
            (
                column
                for column in dataframe.columns
                if column.strip().lower() == "symbol"
            ),
            None,
        )

        if symbol_column is None:
            return []

        symbols = normalize_symbols(
            dataframe[symbol_column]
            .dropna()
            .astype(str)
            .tolist()
        )

        if symbols:
            print(
                f"Loaded {len(symbols)} symbols from local cache.",
                flush=True,
            )

        return symbols

    except Exception as exc:
        print(
            f"Could not read symbol cache: {exc}",
            flush=True,
        )

        return []


def download_official_symbols() -> list[str]:
    """Download the current Nifty 100 list with retries."""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
            "AppleWebKit/605.1.15 Version/17.0 Mobile Safari/604.1"
        ),
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.niftyindices.com/",
        "Connection": "close",
    }

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            print(
                "Downloading official Nifty 100 constituents "
                f"(attempt {attempt}/{DOWNLOAD_RETRIES})...",
                flush=True,
            )

            response = requests.get(
                config.NIFTY100_CSV,
                headers=headers,
                timeout=DOWNLOAD_TIMEOUT,
            )

            response.raise_for_status()

            symbols = parse_constituent_csv(
                response.text
            )

            print(
                f"Downloaded {len(symbols)} official symbols.",
                flush=True,
            )

            return symbols

        except Exception as exc:
            print(
                f"Constituent download attempt {attempt} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            if attempt < DOWNLOAD_RETRIES:
                time.sleep(attempt * 3)

    return []


def nifty100_symbols() -> list[str]:
    """
    Return Nifty 100 Yahoo symbols.

    Priority:
    1. Official Nifty Indices CSV
    2. Local cached CSV
    3. Built-in fallback list
    """

    official_symbols = download_official_symbols()

    if official_symbols:
        save_symbol_cache(official_symbols)
        return official_symbols

    cached_symbols = load_symbol_cache()

    if cached_symbols:
        print(
            "Official constituent download failed. "
            "Using the local cache.",
            flush=True,
        )

        return cached_symbols

    fallback_symbols = normalize_symbols(
        FALLBACK_NIFTY100
    )

    print(
        "Official constituent download and local cache unavailable. "
        f"Using {len(fallback_symbols)} built-in fallback symbols.",
        flush=True,
    )

    return fallback_symbols


def daily(
    symbol: str,
    period: str = "1y",
) -> pd.DataFrame:
    """Download daily price history."""

    try:
        dataframe = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=20,
        )

        if dataframe is None or dataframe.empty:
            return pd.DataFrame()

        return flatten_yf_columns(
            dataframe
        ).dropna()

    except Exception as exc:
        print(
            f"{symbol}: daily download failed: {exc}",
            flush=True,
        )

        return pd.DataFrame()


def intraday(
    symbol: str,
    period: str = "1d",
    interval: str = "5m",
) -> pd.DataFrame:
    """Download intraday price history."""

    try:
        dataframe = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
            prepost=False,
            timeout=20,
        )

        if dataframe is None or dataframe.empty:
            return pd.DataFrame()

        return flatten_yf_columns(
            dataframe
        ).dropna()

    except Exception as exc:
        print(
            f"{symbol}: intraday download failed: {exc}",
            flush=True,
        )

        return pd.DataFrame()


def ticker_info(symbol: str) -> dict[str, Any]:
    """Return Yahoo Finance company information."""

    try:
        information = yf.Ticker(
            symbol
        ).get_info()

        return information or {}

    except Exception as exc:
        print(
            f"{symbol}: information download failed: {exc}",
            flush=True,
        )

        return {}


def recent_news(
    symbol: str,
) -> list[dict[str, Any]]:
    """Return recent Yahoo Finance news when available."""

    try:
        return (
            yf.Ticker(symbol).get_news(
                count=10
            )
            or []
        )

    except Exception:
        return []
