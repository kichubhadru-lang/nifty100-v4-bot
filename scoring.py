from __future__ import annotations
from typing import Any
import config
from adaptive import current_weights


def component_scores(
    metrics: dict[str, Any],
    fundamentals: dict[str, Any],
    market: dict[str, Any],
    news_clear: bool,
) -> dict[str, float]:
    roe = fundamentals["roe"]
    debt = fundamentals["debt_to_equity"]
    margin = fundamentals["profit_margin"]

    fundamental = 0.0
    fundamental += min(40.0, max(0.0, (roe - 0.10) / 0.20 * 40.0))
    fundamental += 30.0 if debt <= 20 else 24.0 if debt <= 35 else 16.0
    fundamental += min(30.0, max(0.0, margin / 0.20 * 30.0))

    drop = abs(metrics["change_pct"])
    technical = 0.0
    technical += 30.0 if 2.5 <= drop <= 4.0 else 20.0 if drop <= 5.0 else 10.0
    technical += 25.0 if metrics["distance_200dma"] >= 0 else 18.0
    technical += max(0.0, 25.0 - abs(metrics["rsi"] - 42.0) * 2.0)
    technical += 20.0 if not metrics["new_52w_low"] else 0.0

    market_score = 100.0
    if market["nifty_change_pct"] < 0:
        market_score -= min(35.0, abs(market["nifty_change_pct"]) * 25.0)
    if not market["nifty_above_50dma"]:
        market_score -= 40.0
    if market["vix"] is not None:
        market_score -= max(0.0, (market["vix"] - 14.0) * 4.0)

    liquidity = min(100.0, metrics["avg_value_20d"] / 2_000_000_000 * 100.0)
    news = 100.0 if news_clear else 0.0

    return {
        "fundamental": round(max(0.0, min(100.0, fundamental)), 2),
        "technical": round(max(0.0, min(100.0, technical)), 2),
        "market": round(max(0.0, min(100.0, market_score)), 2),
        "liquidity": round(max(0.0, min(100.0, liquidity)), 2),
        "news": news,
    }


def weighted_total(components: dict[str, float]) -> tuple[float, dict[str, float]]:
    weights = current_weights()
    total = sum(components[k] * weights[k] / 100.0 for k in weights)
    return round(total, 2), weights
