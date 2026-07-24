from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any
import config
from utils import load_json, save_json


def current_weights() -> dict[str, float]:
    saved = load_json(config.WEIGHTS_FILE, {})
    weights = config.BASE_WEIGHTS.copy()
    for key in weights:
        if isinstance(saved.get(key), (int, float)):
            weights[key] = float(saved[key])
    return normalize(weights)


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        return config.BASE_WEIGHTS.copy()
    return {k: round(max(0.0, v) * 100.0 / total, 4) for k, v in weights.items()}


def bounded(key: str, value: float) -> float:
    base = config.BASE_WEIGHTS[key]
    deviation = base * config.MAX_WEIGHT_DEVIATION_PCT / 100.0
    return min(base + deviation, max(base - deviation, value))


def load_trades() -> list[dict[str, Any]]:
    if not config.TRADE_LOG_FILE.exists():
        return []
    with config.TRADE_LOG_FILE.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def update_weights() -> tuple[dict[str, float], str]:
    trades = load_trades()
    completed = [t for t in trades if t.get("status") == "CLOSED"]
    if len(completed) < config.MIN_COMPLETED_TRADES_TO_LEARN:
        return current_weights(), (
            f"No adaptation: {len(completed)}/"
            f"{config.MIN_COMPLETED_TRADES_TO_LEARN} completed trades."
        )

    sample = completed[-config.LEARNING_WINDOW:]
    contributions: dict[str, list[float]] = defaultdict(list)

    # Each trade stores component scores from 0–100.
    # We measure whether high component scores were associated with better returns.
    for component in config.BASE_WEIGHTS:
        pairs = []
        for trade in sample:
            try:
                score = float(trade[f"{component}_score"])
                ret = float(trade["return_pct"])
                pairs.append((score, ret))
            except (KeyError, TypeError, ValueError):
                continue
        if len(pairs) < config.MIN_BUCKET_TRADES:
            continue

        mean_score = sum(p[0] for p in pairs) / len(pairs)
        high = [r for s, r in pairs if s >= mean_score]
        low = [r for s, r in pairs if s < mean_score]
        if len(high) < 3 or len(low) < 3:
            continue

        # Shrunk edge estimate. The +20 denominator damps small samples.
        raw_edge = (sum(high) / len(high)) - (sum(low) / len(low))
        shrink = len(pairs) / (len(pairs) + 20.0)
        contributions[component].append(raw_edge * shrink)

    old = current_weights()
    proposed = old.copy()
    for key in config.BASE_WEIGHTS:
        edge = contributions.get(key, [0.0])[0]
        # A 1 percentage-point return separation produces only a small weight move.
        proposed[key] = bounded(
            key,
            old[key] * (1.0 + config.LEARNING_RATE * max(-1.0, min(1.0, edge))),
        )

    proposed = normalize(proposed)
    save_json(config.WEIGHTS_FILE, proposed)
    return proposed, f"Adapted from {len(sample)} completed trades with bounded shrinkage."
