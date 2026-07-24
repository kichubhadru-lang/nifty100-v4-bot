from __future__ import annotations
import time
from datetime import datetime
from typing import Any
import pandas as pd
import config
from data_provider import daily, nifty100_symbols, recent_news, ticker_info
from indicators import rsi
from scoring import component_scores, weighted_total
from utils import save_json, telegram_send


def value(series: pd.Series, index: int = -1) -> float:
    return float(series.iloc[index])


def market_regime() -> dict[str, Any]:
    nifty = daily(config.BENCHMARK)
    close = nifty["Close"]
    nifty_change = (value(close) / value(close, -2) - 1.0) * 100.0
    nifty_50dma = float(close.rolling(50).mean().iloc[-1])
    above = value(close) >= nifty_50dma

    vix_value = None
    try:
        vix = daily(config.VIX, period="1mo")
        if not vix.empty:
            vix_value = value(vix["Close"])
    except Exception:
        pass

    allowed = nifty_change > config.MAX_NIFTY_FALL_PCT
    if config.REQUIRE_NIFTY_ABOVE_50DMA:
        allowed = allowed and above
    if vix_value is not None:
        allowed = allowed and vix_value < config.MAX_VIX

    return {
        "allowed": allowed,
        "nifty_change_pct": round(nifty_change, 2),
        "nifty_above_50dma": above,
        "vix": round(vix_value, 2) if vix_value is not None else None,
    }


def blocked_news(symbol: str) -> tuple[bool, list[str]]:
    matched = []
    for item in recent_news(symbol):
        title = str(
            item.get("content", {}).get("title")
            or item.get("title")
            or ""
        ).lower()
        for term in config.BLOCKED_NEWS_TERMS:
            if term in title:
                matched.append(title)
                break
    return bool(matched), matched[:3]


def stock_metrics(symbol: str) -> dict[str, Any] | None:
    df = daily(symbol)
    if len(df) < 220:
        return None

    close = df["Close"]
    volume = df["Volume"]
    latest = value(close)
    previous = value(close, -2)
    change = (latest / previous - 1) * 100
    dma200 = float(close.rolling(200).mean().iloc[-1])
    distance = (latest / dma200 - 1) * 100
    avg_volume = float(volume.rolling(20).mean().iloc[-1])
    avg_value = float((close * volume).rolling(20).mean().iloc[-1])
    volume_ratio = value(volume) / avg_volume if avg_volume else 0.0
    rsi14 = float(rsi(close).iloc[-1])
    low52 = float(df["Low"].tail(252).min())
    new_52w_low = value(df["Low"]) <= low52 * 1.002

    return {
        "symbol": symbol,
        "close": round(latest, 2),
        "change_pct": round(change, 2),
        "distance_200dma": round(distance, 2),
        "avg_value_20d": avg_value,
        "volume_ratio": round(volume_ratio, 2),
        "rsi": round(rsi14, 2),
        "new_52w_low": bool(new_52w_low),
    }


def fundamentals(symbol: str) -> dict[str, Any] | None:
    info = ticker_info(symbol)
    output = {
        "market_cap": info.get("marketCap"),
        "roe": info.get("returnOnEquity"),
        "debt_to_equity": info.get("debtToEquity"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
    }
    required = ("market_cap", "roe", "debt_to_equity", "profit_margin")
    return None if any(output[k] is None for k in required) else output


def passes_rules(m: dict[str, Any], f: dict[str, Any]) -> bool:
    return (
        config.MIN_DROP_PCT <= m["change_pct"] <= config.MAX_DROP_PCT
        and m["close"] >= config.MIN_PRICE
        and m["avg_value_20d"] >= config.MIN_AVG_VALUE_20D
        and config.MIN_DISTANCE_200DMA <= m["distance_200dma"] <= config.MAX_DISTANCE_200DMA
        and config.MIN_RSI <= m["rsi"] <= config.MAX_RSI
        and m["volume_ratio"] >= config.MIN_VOLUME_RATIO
        and not m["new_52w_low"]
        and f["market_cap"] >= config.MIN_MARKET_CAP
        and f["roe"] >= config.MIN_ROE
        and f["debt_to_equity"] <= config.MAX_DEBT_TO_EQUITY_YAHOO
        and f["profit_margin"] > config.MIN_PROFIT_MARGIN
    )


def run_scan() -> None:
    market = market_regime()
    if not market["allowed"]:
        save_json(config.WATCHLIST_FILE, {"date": str(datetime.now().date()), "candidates": []})
        telegram_send(
            "⛔ <b>V4: No trade regime</b>\n"
            f"Nifty: {market['nifty_change_pct']}%\n"
            f"Above 50-DMA: {market['nifty_above_50dma']}\n"
            f"India VIX: {market['vix']}"
        )
        return

    candidates = []
    errors = []
    for symbol in nifty100_symbols():
        try:
            m = stock_metrics(symbol)
            if m is None:
                continue
            # Cheap filters before slower fundamental/news calls.
            if not (
                config.MIN_DROP_PCT <= m["change_pct"] <= config.MAX_DROP_PCT
                and m["avg_value_20d"] >= config.MIN_AVG_VALUE_20D
                and config.MIN_RSI <= m["rsi"] <= config.MAX_RSI
            ):
                continue

            f = fundamentals(symbol)
            if f is None or not passes_rules(m, f):
                continue

            is_blocked, headlines = blocked_news(symbol)
            components = component_scores(m, f, market, not is_blocked)
            total, weights = weighted_total(components)
            if is_blocked or total < config.MIN_SIGNAL_SCORE:
                continue

            candidates.append({
                **m,
                **f,
                "component_scores": components,
                "score": total,
                "weights": weights,
                "blocked_headlines": headlines,
            })
            time.sleep(0.2)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:config.MAX_CANDIDATES]
    save_json(
        config.WATCHLIST_FILE,
        {
            "date": str(datetime.now().date()),
            "market": market,
            "candidates": candidates,
            "errors": errors[:10],
        },
    )

    if not candidates:
        telegram_send(
            "✅ <b>V4 scan completed</b>\n"
            "No Nifty 100 stock passed every rule.\n"
            f"Scanner errors: {len(errors)}"
        )
        return

    lines = ["📉 <b>V4 Nifty 100 Watchlist</b>", ""]
    for index, c in enumerate(candidates, 1):
        cs = c["component_scores"]
        lines.extend([
            f"<b>{index}. {c['symbol'].replace('.NS', '')}</b> — {c['score']}/100",
            f"Close ₹{c['close']} | Fall {c['change_pct']}%",
            f"RSI {c['rsi']} | Volume {c['volume_ratio']}×",
            f"F {cs['fundamental']} | T {cs['technical']} | M {cs['market']}",
            "",
        ])
    lines.append("Entry is not active. Wait for next-day 15-minute confirmation.")
    telegram_send("\n".join(lines))
