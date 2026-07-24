from __future__ import annotations
import csv
from statistics import mean, median
import config
from adaptive import current_weights, update_weights
from utils import telegram_send


def run_report(adapt: bool = False) -> None:
    if not config.TRADE_LOG_FILE.exists():
        telegram_send("📊 V4 report: no completed paper trades yet.")
        return

    with config.TRADE_LOG_FILE.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "CLOSED"]

    returns = [float(r["return_pct"]) for r in rows]
    if not returns:
        telegram_send("📊 V4 report: no completed paper trades yet.")
        return

    wins = sum(r > 0 for r in returns)
    gross_win = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    profit_factor = gross_win / gross_loss if gross_loss else float("inf")

    message = [
        "📊 <b>V4 PERFORMANCE REPORT</b>",
        f"Trades: {len(returns)}",
        f"Win rate: {wins / len(returns) * 100:.2f}%",
        f"Average: {mean(returns):.3f}%",
        f"Median: {median(returns):.3f}%",
        f"Profit factor: {profit_factor:.2f}",
    ]

    if adapt:
        weights, note = update_weights()
        message.extend(["", f"Learning: {note}", f"Weights: {weights}"])
    else:
        message.extend(["", f"Current weights: {current_weights()}"])

    telegram_send("\n".join(message))
